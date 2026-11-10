"""Fetch DTEK outages, format reports, and track scheduled notifications."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from html import escape
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from core.logger import logger
from services.electricity_status import ElectricityChecker, ElectricityStatus

DTEK_URL = "https://www.dtek-krem.com.ua/ua/shutdowns"
DTEK_AJAX_URL = "https://www.dtek-krem.com.ua/ua/ajax"
AJAX_TIMEOUT_SECONDS = 15
SCRIPT_TIMEOUT_SECONDS = AJAX_TIMEOUT_SECONDS + 5

DTEK_TIMEZONE = ZoneInfo("Europe/Kyiv")


def _dtek_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=DTEK_TIMEZONE)
    return value.astimezone(DTEK_TIMEZONE)


# ============================================================================
# MODELS
# ============================================================================
class MonitorContext(BaseModel):
    """Holds configuration to avoid global variables"""

    city: str
    street: str
    building: str
    forced_group: str | None
    state_file: Path

    @field_validator("city", mode="before")
    def parse_city(cls, city: str) -> str:
        return f"м.+{city}" if not city.startswith("м.+") else city

    @field_validator("street", mode="before")
    def parse_street(cls, street: str) -> str:
        return f"вул.+{street}" if not street.startswith("вул.+") else street


class TimeType(str, Enum):
    YES = "yes"
    MAYBE = "maybe"
    NO = "no"
    FIRST = "first"
    SECOND = "second"
    MFIRST = "mfirst"
    MSECOND = "msecond"


class OutagePeriod(BaseModel):
    start: datetime
    end: datetime
    type: str  # "no" or "maybe"

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

    def format_time_range(self) -> str:
        """Format period as time range string"""
        return f"{self.start.strftime('%H:%M')} - {self.end.strftime('%H:%M')}"

    def get_icon(self) -> str:
        """Get icon based on outage type"""
        return "❔" if self.type == "maybe" else "❌"


class DaySchedule(BaseModel):
    date: datetime
    periods: list[OutagePeriod] = Field(default_factory=list)
    updated_at: datetime | None = None

    def format_date(self) -> str:
        """Format date as weekday and date"""
        return self.date.strftime("%a, %d.%m")


class CurrentOutage(BaseModel):
    sub_type: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    group: str | None = None
    updated_at: datetime | None = None

    @field_validator("start_date", "end_date", "updated_at", mode="before")
    def parse_dates(cls, v: str | bool | None) -> datetime | None:
        if not v:
            return None
        if isinstance(v, datetime):
            return _dtek_time(v)
        if isinstance(v, str):
            # DTEK format (H:M d.m.Y from API)
            try:
                return datetime.strptime(v.strip(), "%H:%M %d.%m.%Y").replace(
                    tzinfo=DTEK_TIMEZONE
                )
            except ValueError:
                pass
            # ISO format (from state file)
            try:
                return _dtek_time(datetime.fromisoformat(v.strip()))
            except ValueError:
                return None

        return None

    @property
    def is_active(self) -> bool:
        """Returns True if DTEK says the lights are currently out"""
        return bool(self.start_date and self.end_date)

    def format_time_range(self) -> str:
        """Format outage time range"""
        if self.start_date and self.end_date:
            return f"🕒 {self.start_date.strftime('%H:%M')} - {self.end_date.strftime('%H:%M')}"
        return ""

    def __eq__(self, other) -> bool:
        """Compare two outage statuses"""
        if not isinstance(other, CurrentOutage):
            return False
        # Only compare critical fields
        return (
            self.start_date == other.start_date
            and self.end_date == other.end_date
            and self.sub_type == other.sub_type
        )


# ============================================================================
# DATA FETCHER
# ============================================================================
class DTEKMonitor:
    """Monitors DTEK power outage information"""

    def _init_driver(self):
        """Configure and return Chrome WebDriver"""
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

        try:
            driver = webdriver.Chrome(options=options)
            return driver
        except WebDriverException as e:
            logger.critical(f"Failed to start WebDriver: {e}")
            raise

    def _extract_schedule_var(self, html: str) -> dict | None:
        """Regex extraction of the schedule variable embedded in HTML"""
        pattern = r"DisconSchedule\.fact\s*=\s*(\{[^<]+?\})\s*(?:</script>|DisconSchedule\.|var\s+|$)"
        match = re.search(pattern, html, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.exception("Failed to parse schedule JSON")
        return None

    def fetch(self, city: str, street: str) -> tuple[dict | None, dict | None]:
        """Orchestrates the fetching of both current status (AJAX) and schedule (JS Var)"""
        driver = None
        stage = "browser startup"
        try:
            driver = self._init_driver()
            stage = "page load"
            driver.set_page_load_timeout(30)

            logger.info(f"Loading page: {DTEK_URL}")
            driver.get(DTEK_URL)
            driver.implicitly_wait(5)
            driver.find_element(By.TAG_NAME, "script")

            # Scrape Schedule Variable
            page_source = driver.page_source
            schedule_json = self._extract_schedule_var(page_source)

            # AJAX for Current Status
            stage = "CSRF token lookup"
            csrf_token = driver.execute_script(
                "return document.querySelector('meta[name=\"csrf-token\"]').content;"
            )

            stage = "AJAX request"
            logger.info(f"Fetching AJAX data: {DTEK_AJAX_URL}")
            form_body = urlencode(
                [
                    ("method", "getHomeNum"),
                    ("data[0][name]", "city"),
                    ("data[0][value]", city.replace("+", " ")),
                    ("data[1][name]", "street"),
                    ("data[1][value]", street.replace("+", " ")),
                ]
            )
            js_payload = """
            const callback = arguments[arguments.length - 1];
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), arguments[3]);
            fetch(arguments[0], {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                    "X-Requested-With": "XMLHttpRequest",
                    "X-CSRF-Token": arguments[1]
                },
                body: arguments[2],
                signal: controller.signal
            }).then(async response => {
                if (!response.ok) {
                    return {kind: "http_error", status: response.status};
                }
                try {
                    return {kind: "ok", status: response.status, data: await response.json()};
                } catch (error) {
                    if (error.name === "AbortError") {
                        return {kind: "timeout", error_name: "AbortError"};
                    }
                    return {
                        kind: "invalid_json", status: response.status,
                        content_type: response.headers.get("content-type")
                    };
                }
            }).catch(error => ({
                kind: error.name === "AbortError" ? "timeout" : "network_error",
                error_name: error.name || "Error",
                error_message: String(error.message || "").slice(0, 200)
            })).then(result => {
                clearTimeout(timer);
                callback(result);
            });
            """
            driver.set_script_timeout(SCRIPT_TIMEOUT_SECONDS)
            started = time.monotonic()
            ajax_result = driver.execute_async_script(
                js_payload,
                DTEK_AJAX_URL,
                csrf_token,
                form_body,
                AJAX_TIMEOUT_SECONDS * 1000,
            )
            elapsed = time.monotonic() - started
            if not isinstance(ajax_result, dict):
                logger.warning(
                    "DTEK AJAX returned an invalid script result (%s) after %.1fs",
                    type(ajax_result).__name__,
                    elapsed,
                )
                return None, None

            kind = ajax_result.get("kind")
            if kind != "ok":
                logger.warning(
                    "DTEK AJAX failed: %s (HTTP %s, %s, %.1fs)",
                    kind or "unknown_error",
                    ajax_result.get("status", "none"),
                    ajax_result.get("error_message")
                    or ajax_result.get("error_name")
                    or ajax_result.get("content_type")
                    or "no detail",
                    elapsed,
                )
                return None, None

            outage_json = ajax_result.get("data")
            if not isinstance(outage_json, dict):
                logger.warning(
                    "DTEK AJAX JSON root is %s (HTTP %s, %.1fs)",
                    type(outage_json).__name__,
                    ajax_result.get("status", "unknown"),
                    elapsed,
                )
                return None, None
            if not outage_json.get("result"):
                logger.warning(
                    "DTEK AJAX JSON result is %s (HTTP %s, %.1fs)",
                    "missing" if "result" not in outage_json else "false",
                    ajax_result.get("status", "unknown"),
                    elapsed,
                )
                return None, None
            logger.info(
                "DTEK AJAX succeeded (HTTP %s, %.1fs)",
                ajax_result.get("status"),
                elapsed,
            )
            return outage_json, schedule_json

        except TimeoutException:
            logger.exception("DTEK %s exceeded its timeout", stage)
            return None, None
        except WebDriverException:
            logger.exception("DTEK browser error during %s", stage)
            return None, None
        finally:
            if driver:
                try:
                    driver.quit()
                except WebDriverException as e:
                    logger.warning(f"Failed to close WebDriver: {e}")


# ============================================================================
# MESSAGE FORMATTER
# ============================================================================
class MessageFormatter:
    """Handles all message formatting logic"""

    def __init__(self, ctx: MonitorContext):
        self.ctx = ctx

    def format_header(self, outage: CurrentOutage) -> list[str]:
        """Generate header section with location and group info"""
        lines = []
        location = (
            f"{self.ctx.city.replace('+', ' ')}, "
            f"{self.ctx.street.replace('+', ' ')}, {self.ctx.building}"
        )
        lines.append(f"📍 <b>{escape(location)}</b>")

        group = self.ctx.forced_group or outage.group
        if group:
            lines.append(f"  Черга: <b>{escape(group)}</b>")

        return lines

    def format_alert_status(
        self,
        electricity_status: ElectricityStatus,
        upcoming: list[datetime],
        now: datetime,
    ) -> list[str]:
        """Generate alert section for active outages or upcoming warnings"""
        lines = []

        if electricity_status == ElectricityStatus.OUTAGE:
            lines.append("\n🔴 <b>Світло відсутнє!</b>")
        for start in upcoming:
            mins = max(0, int((start - now).total_seconds() / 60))
            lines.append(f"\n⚠️ <b>Увага! Відключення через {mins} хв</b>")

        return lines

    def format_current_status(
        self, electricity_status: ElectricityStatus, outage: CurrentOutage
    ) -> list[str]:
        """Generate the live power status with available DTEK context."""
        lines = ["\n📌 <b>Поточний стан електропостачання</b>"]
        status_messages = {
            ElectricityStatus.OK: "✅ Зараз світло є",
            ElectricityStatus.OUTAGE: "🔴 Зараз світла немає",
            ElectricityStatus.UNKNOWN: "⚫️ Не вдалося перевірити поточний стан",
        }
        lines.append(status_messages[electricity_status])

        if outage.sub_type:
            lines.append(f"Причина за даними ДТЕК: {escape(outage.sub_type)}")
        time_range = outage.format_time_range()
        if time_range:
            lines.append(f"Час за даними ДТЕК (орієнтовно): {time_range}")

        return lines

    def format_unavailable_report(
        self, electricity_status: ElectricityStatus, outage: CurrentOutage
    ) -> str:
        """Show live power status when the DTEK schedule cannot be read."""
        if electricity_status == ElectricityStatus.UNKNOWN:
            return "⚫️ Не вдалося перевірити поточний стан і графік ДТЕК"
        lines = self.format_header(outage)
        if electricity_status == ElectricityStatus.OUTAGE:
            lines.append("\n🔴 <b>Світло відсутнє!</b>")
        lines.extend(self.format_current_status(electricity_status, outage))
        lines.append("\n📅 <b>Графік відключень</b>")
        lines.append("⚠️ Графік ДТЕК недоступний")
        return "\n".join(lines)

    def format_schedule(self, schedules: list[DaySchedule]) -> list[str]:
        """Generate schedule section with daily outage periods"""
        lines = ["\n📅 <b>Графік відключень</b>"]

        if not schedules:
            lines.append("🔋 Без відключень")
            return lines

        for day in schedules:
            lines.append(f"<b>{day.format_date()}</b>:")

            if not day.periods:
                lines.append("  🔋 Без відключень")
            else:
                for period in day.periods:
                    lines.append(f"  {period.get_icon()} {period.format_time_range()}")

        if schedules and schedules[0].updated_at:
            lines.append(
                f"<i>Графік оновлено:</i> {schedules[0].updated_at.strftime('%H:%M')}"
            )

        return lines

    def generate_report(
        self,
        electricity_status: ElectricityStatus,
        outage: CurrentOutage,
        schedules: list[DaySchedule],
        upcoming: list[datetime],
        now: datetime,
    ) -> str:
        """Generate complete human-readable report"""
        lines = []

        lines.extend(self.format_header(outage))
        lines.extend(self.format_alert_status(electricity_status, upcoming, now))
        lines.extend(self.format_current_status(electricity_status, outage))
        lines.extend(self.format_schedule(schedules))

        return "\n".join(lines)


# ============================================================================
# SCHEDULE PARSER
# ============================================================================
class DayScheduleParser:
    """Parses schedule for a single day"""

    def __init__(
        self,
        base_date: datetime,
        hours_map: dict[str, str],
        updated_at: datetime | None,
    ):
        self.base_date = base_date
        self.hours_map = hours_map
        self.updated_at = updated_at
        self.periods: list[OutagePeriod] = []
        self.current_period_start: datetime | None = None
        self.current_period_type: str = "yes"

    def parse(self) -> DaySchedule:
        """Parse all hours for this day"""
        sorted_hours = sorted(int(k) for k in self.hours_map)

        for hour in sorted_hours:
            status = TimeType(self.hours_map[str(hour)])
            self._process_hour(hour, status)

        self._finalize_periods()

        return DaySchedule(
            date=self.base_date,
            periods=self.periods,
            updated_at=self.updated_at,
        )

    def _process_hour(self, hour: int, status: TimeType):
        """Process a single hour based on its status"""

        # Determine start and end of this specific hour slot
        slot_start = self._create_time(hour - 1, 0)
        slot_mid = self._create_time(hour - 1, 30)

        # Handle full hour outage (NO or MAYBE)
        if status in [TimeType.NO, TimeType.MAYBE]:
            self._start_period_if_needed(slot_start, status.value)

        # Handle first half hour outage (FIRST or MFIRST)
        elif status in [TimeType.FIRST, TimeType.MFIRST]:
            outage_type = "maybe" if status == TimeType.MFIRST else "no"
            period_start = self._start_period_if_needed(slot_start, outage_type)

            self._add_period(
                start=period_start,
                end=slot_mid,
                outage_type=self.current_period_type,
            )
            self.current_period_start = None

        # Handle second half hour outage (SECOND or MSECOND)
        elif status in [TimeType.SECOND, TimeType.MSECOND]:
            outage_type = "maybe" if status == TimeType.MSECOND else "no"
            if self.current_period_start:
                self._add_period(
                    start=self.current_period_start,
                    end=slot_start,
                    outage_type=self.current_period_type,
                )
            self.current_period_start = slot_mid
            self.current_period_type = outage_type

        # End the current outage period
        else:  # YES (Power ON)
            if self.current_period_start:
                self._add_period(
                    start=self.current_period_start,
                    end=slot_start,
                    outage_type=self.current_period_type,
                )
                self.current_period_start = None

    def _start_period_if_needed(self, start: datetime, outage_type: str) -> datetime:
        if self.current_period_start is None:
            self.current_period_start = start
            self.current_period_type = outage_type
            return start
        return self.current_period_start

    def _finalize_periods(self):
        """Finalize any remaining period at end of day"""
        if self.current_period_start:
            final_end = self._create_time(24, 0)
            self._add_period(
                start=self.current_period_start,
                end=final_end,
                outage_type=self.current_period_type,
            )

    def _add_period(self, start: datetime, end: datetime, outage_type: str):
        """Add an outage period to the list"""
        self.periods.append(OutagePeriod(start=start, end=end, type=outage_type))

    def _create_time(self, hour: int, minute: int) -> datetime:
        """Create datetime handling special cases like hour=24"""
        if hour == 24:
            return (self.base_date + timedelta(days=1)).replace(
                hour=0, minute=minute, second=0, microsecond=0
            )
        if hour == -1:
            return (self.base_date - timedelta(days=1)).replace(
                hour=23, minute=minute, second=0, microsecond=0
            )
        return self.base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)


class ScheduleParser:
    """Handles parsing of schedule data from DTEK API"""

    def __init__(self, data: dict | None, group: str | None):
        self.data = data
        self.group = group
        self.updated_at = self._parse_update_timestamp()

    def _parse_update_timestamp(self) -> datetime | None:
        """Parse the update timestamp from schedule data"""
        if not isinstance(self.data, dict) or not self.data.get("update"):
            return None
        try:
            # DTEK timestamp format: d.m.Y H:M
            return datetime.strptime(self.data["update"], "%d.%m.%Y %H:%M").replace(
                tzinfo=DTEK_TIMEZONE
            )
        except TypeError, ValueError:
            return None

    def _schedule_pairs(self) -> list[tuple[object, object]]:
        """Normalize the old and new API schedule formats."""
        if not isinstance(self.data, dict) or "data" not in self.data:
            raise ValueError("Schedule data is missing")

        raw = self.data["data"]

        if isinstance(raw, dict):
            return list(raw.items())
        if isinstance(raw, list):
            pairs = []
            for item in raw:
                if not isinstance(item, dict) or not item:
                    raise ValueError("Schedule day data is invalid")
                pairs.extend(item.items())
            return pairs
        raise TypeError(f"Unexpected schedule data type: {type(raw).__name__}")

    def parse(self) -> list[DaySchedule]:
        """Parse schedule data for the configured group"""
        pairs = self._schedule_pairs()
        if not pairs:
            return []
        if not self.group:
            raise ValueError("Schedule group is missing")

        group_key = f"GPV{self.group}"
        schedules = []

        for timestamp, groups_data in pairs:
            try:
                if not isinstance(groups_data, dict) or group_key not in groups_data:
                    raise ValueError(f"No data for group {group_key}")
                if not isinstance(timestamp, (str, int, float)):
                    raise TypeError("Schedule timestamp is invalid")
                base_date = datetime.fromtimestamp(int(timestamp), tz=DTEK_TIMEZONE)
                hours_map = groups_data[group_key]

                day_parser = DayScheduleParser(base_date, hours_map, self.updated_at)
                schedule = day_parser.parse()
                schedules.append(schedule)
            except Exception as e:
                raise ValueError(f"Error parsing schedule for {timestamp}: {e}") from e

        return schedules


# ============================================================================
# STATE MANAGEMENT
# ============================================================================
class StateManager:
    """Stores per-chat delivery state and excludes overlapping scheduled runs."""

    @staticmethod
    def load(file_path: Path) -> NotificationState:
        if not file_path.exists():
            return NotificationState()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or data.get("version") != 2:
                logger.info("Replacing previous DTEK notification state format")
                return NotificationState()
            return NotificationState.model_validate(data)
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to load state: {e}")
            return NotificationState()

    @staticmethod
    def save(file_path: Path, state: NotificationState) -> None:
        """Replace state atomically so a stopped process cannot truncate it."""
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=file_path.parent,
                prefix=f".{file_path.name}.",
                delete=False,
            ) as f:
                temporary_path = Path(f.name)
                json.dump(
                    state.model_dump(mode="json"), f, ensure_ascii=False, indent=2
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, file_path)
        except Exception:
            logger.exception("Failed to save state")
            raise
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    @contextmanager
    def lock(file_path: Path) -> Generator[bool]:
        lock_path = file_path.with_name(f"{file_path.name}.lock")
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


class ChatState(BaseModel):
    outage: CurrentOutage
    schedule_periods: str
    warned_starts: set[str] = Field(default_factory=set)


class NotificationState(BaseModel):
    version: int = 2
    chats: dict[str, ChatState] = Field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    outage: CurrentOutage
    schedule_periods: str
    upcoming: list[datetime]
    message: str
    observed_at: datetime
    suppressed_upcoming: list[datetime] = field(default_factory=list)


@dataclass(frozen=True)
class ChatPlan:
    """What to do for one chat after an observation.

    baseline: store next_state now, send nothing.
    silent: nothing changed, store nothing.
    alert: send message, and store next_state only once delivery is accepted.
    """

    kind: Literal["baseline", "silent", "alert"]
    next_state: ChatState
    message: str = ""


def _days_from(periods_json: str, today: datetime) -> list[dict]:
    """Schedule days dated today or later; midnight rollover must not look like a change."""
    try:
        days = json.loads(periods_json)
    except json.JSONDecodeError:
        return []
    today_key = today.strftime("%Y-%m-%d")
    return [
        day
        for day in days
        if not isinstance(day, dict) or day.get("date", "") >= today_key
    ]


def plan_chat(previous: ChatState | None, observation: Observation) -> ChatPlan:
    """Decide one chat's scheduled alert from its last acknowledged state."""
    upcoming_keys = {start.isoformat() for start in observation.upcoming}
    suppressed_keys = {start.isoformat() for start in observation.suppressed_upcoming}
    cutoff = (observation.observed_at - timedelta(days=1)).isoformat()
    warned = previous.warned_starts if previous else set()
    next_state = ChatState(
        outage=observation.outage,
        schedule_periods=observation.schedule_periods,
        warned_starts={key for key in warned if key >= cutoff}
        | upcoming_keys
        | suppressed_keys,
    )
    if previous is None:
        return ChatPlan("baseline", next_state)

    reasons = []
    if previous.outage != observation.outage:
        reasons.append("ЗМІНА ВІДКЛЮЧЕННЯ")
    today = observation.observed_at
    if _days_from(previous.schedule_periods, today) != _days_from(
        observation.schedule_periods, today
    ):
        reasons.append("ЗМІНА ГРАФІКА")
    if upcoming_keys - warned:
        reasons.append("НАБЛИЖАЄТЬСЯ ВІДКЛЮЧЕННЯ")
    if not reasons:
        return ChatPlan("silent", next_state)

    headers = "\n".join(f"<code>{reason}</code>" for reason in reasons)
    return ChatPlan("alert", next_state, f"{headers}\n\n{observation.message}")


# ============================================================================
# MONITOR SERVICE
# ============================================================================
class MonitorService:
    """Owns DTEK reports and per-chat notification decisions."""

    def __init__(
        self,
        ctx: MonitorContext,
        electricity: ElectricityChecker,
        source: DTEKMonitor | None = None,
        state: StateManager | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.ctx = ctx
        self.electricity = electricity
        self.source = source if source is not None else DTEKMonitor()
        self.state = state if state is not None else StateManager()
        self.clock = clock if clock is not None else lambda: datetime.now(DTEK_TIMEZONE)
        self.formatter = MessageFormatter(ctx)

    def _observe(self) -> tuple[Observation | None, str]:
        electricity_status = self.electricity.check().status
        outage = CurrentOutage()
        try:
            outage_data, schedule_data = self.source.fetch(
                city=self.ctx.city, street=self.ctx.street
            )
            if not outage_data:
                return None, self.formatter.format_unavailable_report(
                    electricity_status, outage
                )
            outage = parse_current_outage(outage_data, self.ctx.building)
            group = self.ctx.forced_group or outage.group
            schedules = ScheduleParser(schedule_data, group).parse()
            now = _dtek_time(self.clock())
            ends = {(p.end, p.type) for day in schedules for p in day.periods}
            # A start equal to another period's end is the same outage split at midnight
            starts = sorted(
                period.start
                for day in schedules
                for period in day.periods
                if (period.start, period.type) not in ends
                and now < period.start <= now + timedelta(minutes=25)
            )
            # No warning while power is already off; starts are still marked as warned
            power_off = electricity_status == ElectricityStatus.OUTAGE
            upcoming = [] if power_off else starts
            message = self.formatter.generate_report(
                electricity_status, outage, schedules, upcoming, now
            )
            return Observation(
                outage=outage,
                schedule_periods=get_schedule_periods_json(schedules),
                upcoming=upcoming,
                message=message,
                observed_at=now,
                suppressed_upcoming=starts if power_off else [],
            ), ""
        except ValueError, TypeError, WebDriverException:
            logger.exception("DTEK observation failed")
            return None, self.formatter.format_unavailable_report(
                electricity_status, outage
            )

    def report(self) -> str:
        """Fetch a report without reading or changing notification state."""
        observation, error = self._observe()
        return observation.message if observation else error

    def notify(
        self, chat_ids: tuple[str, ...], send: Callable[[str, str], bool]
    ) -> str | None:
        """Run scheduled monitoring; return None when another run holds the lock."""
        if not chat_ids:
            return self.report()
        with self.state.lock(self.ctx.state_file) as acquired:
            if not acquired:
                logger.info("Overlapping DTEK monitoring run skipped")
                return None
            state = self.state.load(self.ctx.state_file)
            self.state.save(self.ctx.state_file, state)
            observation, error = self._observe()
            if observation is None:
                logger.warning("DTEK monitoring failed; scheduled report skipped")
                return error
            self._notify_report(state, chat_ids, send, observation)
            return observation.message

    def _send(
        self, send: Callable[[str, str], bool], chat_id: str, message: str
    ) -> bool:
        try:
            return bool(send(chat_id, message))
        except requests.RequestException:
            logger.exception("Failed to send DTEK alert to %s", chat_id)
            return False

    def _notify_report(
        self,
        state: NotificationState,
        chat_ids: tuple[str, ...],
        send: Callable[[str, str], bool],
        observation: Observation,
    ) -> None:
        for chat_id in dict.fromkeys(chat_ids):
            plan = plan_chat(state.chats.get(chat_id), observation)
            if plan.kind == "baseline":
                state.chats[chat_id] = plan.next_state
                self.state.save(self.ctx.state_file, state)
                logger.info("DTEK baseline saved for chat %s; no message sent", chat_id)
            elif plan.kind == "silent":
                if plan.next_state.warned_starts != state.chats[chat_id].warned_starts:
                    # Remember suppressed warnings so they are not sent late
                    state.chats[chat_id] = plan.next_state
                    self.state.save(self.ctx.state_file, state)
                logger.info(
                    "DTEK report unchanged for chat %s; no message sent", chat_id
                )
            elif self._send(send, chat_id, plan.message):
                state.chats[chat_id] = plan.next_state
                self.state.save(self.ctx.state_file, state)


# ============================================================================
# UTILITIES
# ============================================================================
def parse_current_outage(data: dict, building: str) -> CurrentOutage:
    """Parse raw outage data"""

    if not data or not data.get("result"):
        logger.warning("No outage data result found")
        return CurrentOutage()

    raw_data = data.get("data", {})
    item = raw_data.get(building)
    if not item:
        logger.warning(f"Building {building} not found in response")
        return CurrentOutage()

    group = None
    if item.get("sub_type_reason"):
        raw_reason = item["sub_type_reason"][0]
        group = raw_reason.replace("GPV", "").strip()

    return CurrentOutage(
        sub_type=item.get("sub_type"),
        start_date=item.get("start_date"),
        end_date=item.get("end_date"),
        group=group,
        updated_at=data.get("updateTimestamp"),
    )


def get_schedule_periods_json(schedules: list[DaySchedule]) -> str:
    """Generates JSON string for comparison, based only on date and periods"""

    data = []
    for day in schedules:
        day_data = {
            "date": day.date.strftime("%Y-%m-%d"),
            "periods": [
                {
                    "start": p.start.strftime("%H:%M"),
                    "end": p.end.strftime("%H:%M"),
                    "type": p.type,
                }
                for p in day.periods
            ],
        }
        data.append(day_data)

    return json.dumps(data, sort_keys=True, ensure_ascii=False)
