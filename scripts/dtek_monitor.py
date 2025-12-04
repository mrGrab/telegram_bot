#!/usr/bin/env python3
# coding: utf-8
"""
Power Outage Monitoring Script for DTEK
Monitors scheduled power outages and sends notifications via Telegram
"""

import os
import re
import json
import click
import logging
import requests
from enum import Enum
from pathlib import Path
from rich.console import Console
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List, Tuple, Union

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

DTEK_URL = "https://www.dtek-krem.com.ua/ua/shutdowns"
DTEK_AJAX_URL = "https://www.dtek-krem.com.ua/ua/ajax"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================
class MonitorContext(BaseModel):
    """Holds configuration to avoid global variables"""
    city: str
    street: str
    building: str
    forced_group: Optional[str]
    state_file: Path

    @field_validator('city', mode='before')
    def parse_city(cls, city: str) -> str:
        return f"м.+{city}" if not city.startswith("м.+") else city

    @field_validator('street', mode='before')
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
    periods: List[OutagePeriod] = Field(default_factory=list)
    updated_at: Optional[datetime] = None

    def get_upcoming_outage(self, minutes_ahead: int) -> Optional[datetime]:
        """Check if outage starts within specified minutes"""
        now = datetime.now()

        # Only check today's schedule
        if self.date.date() != now.date():
            return None

        for period in self.periods:
            # Check if period starts in the future but within the window
            time_until = (period.start - now).total_seconds() / 60
            if 0 < time_until <= minutes_ahead:
                return period.start
        return None

    def format_date(self) -> str:
        """Format date as weekday and date"""
        return self.date.strftime("%a, %d.%m")


class CurrentOutage(BaseModel):
    sub_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    group: Optional[str] = None
    updated_at: Optional[datetime] = None

    @field_validator('start_date', 'end_date', 'updated_at', mode='before')
    def parse_dates(cls, v: Union[str, bool, None]) -> Optional[datetime]:
        if not v:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # DTEK format (H:M d.m.Y from API)
            try:
                return datetime.strptime(v.strip(), "%H:%M %d.%m.%Y")
            except ValueError:
                pass
            # ISO format (from state file)
            try:
                return datetime.fromisoformat(v.strip())
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
        return (self.start_date == other.start_date
                and self.end_date == other.end_date
                and self.sub_type == other.sub_type)


# ============================================================================
# DATA FETCHER
# ============================================================================
class DTEKMonitor:
    """Monitors DTEK power outage information"""

    def __init__(self):
        self.driver: webdriver.Chrome

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
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(30)
        except WebDriverException as e:
            logger.critical(f"Failed to start WebDriver: {e}")
            raise

    def _extract_schedule_var(self, html: str) -> Optional[Dict]:
        """Regex extraction of the schedule variable embedded in HTML"""
        pattern = r'DisconSchedule\.fact\s*=\s*(\{[^<]+?\})\s*(?:</script>|DisconSchedule\.|var\s+|$)'
        match = re.search(pattern, html, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse schedule JSON: {e}")
        return None

    def fetch(self, city: str,
              street: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Orchestrates the fetching of both current status (AJAX) and schedule (JS Var)"""
        try:
            self._init_driver()
            logger.info(f"Loading page: {DTEK_URL}")
            self.driver.get(DTEK_URL)
            self.driver.implicitly_wait(5)
            self.driver.find_element(By.TAG_NAME, "script")

            # Scrape Schedule Variable
            page_source = self.driver.page_source
            schedule_json = self._extract_schedule_var(page_source)

            # AJAX for Current Status
            csrf_token = self.driver.execute_script(
                'return document.querySelector(\'meta[name="csrf-token"]\').content;'
            )

            logger.info(f"Fetching AJAX data: {DTEK_AJAX_URL}")
            js_payload = f"""
            return fetch("{DTEK_AJAX_URL}", {{
                method: "POST",
                headers: {{
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                    "X-Requested-With": "XMLHttpRequest",
                    "X-CSRF-Token": "{csrf_token}"
                }},
                body: "method=getHomeNum&data[0][name]=city&data[0][value]={city}&data[1][name]=street&data[1][value]={street}"
            }}).then(r => r.json());
            """
            outage_json = self.driver.execute_script(js_payload)
            return outage_json, schedule_json

        except Exception as e:
            logger.error(f"Browser error: {e}")
            return None, None
        finally:
            if self.driver:
                self.driver.quit()


# ============================================================================
# MESSAGE FORMATTER
# ============================================================================
class MessageFormatter:
    """Handles all message formatting logic"""

    def __init__(self, ctx: MonitorContext):
        self.ctx = ctx

    def format_header(self, outage: CurrentOutage) -> List[str]:
        """Generate header section with location and group info"""
        lines = []
        lines.append(f"📍 <b>{self.ctx.city.replace('+', ' ')}, "
                     f"{self.ctx.street.replace('+', ' ')}, "
                     f"{self.ctx.building}</b>")

        group = self.ctx.forced_group or outage.group
        if group:
            lines.append(f"  Черга: <b>{group}</b>")

        return lines

    def format_alert_status(self, outage: CurrentOutage,
                            upcoming: Optional[datetime]) -> List[str]:
        """Generate alert section for active outages or upcoming warnings"""
        lines = []

        if outage.is_active:
            lines.append("\n🔴 <b>Світло відсутнє!</b>")
        elif upcoming:
            mins = int((upcoming - datetime.now()).total_seconds() / 60)
            lines.append(f"\n⚠️ <b>Увага! Відключення через {mins} хв</b>")

        return lines

    def format_current_status(self, outage: CurrentOutage) -> List[str]:
        """Generate current outage status section"""
        lines = ["\n📌 <b>Поточне відключення</b>"]

        if outage.is_active:
            if outage.sub_type:
                lines.append(f"{outage.sub_type}")
            time_range = outage.format_time_range()
            if time_range:
                lines.append(time_range)
        else:
            lines.append("✅ Зараз відключень немає")

        if outage.updated_at:
            lines.append(
                f"<i>Оновлено:</i> {outage.updated_at.strftime('%H:%M')}")

        return lines

    def format_schedule(self, schedules: List[DaySchedule]) -> List[str]:
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
                    lines.append(
                        f"  {period.get_icon()} {period.format_time_range()}")

        if schedules and schedules[0].updated_at:
            lines.append(
                f"<i>Графік оновлено:</i> {schedules[0].updated_at.strftime('%H:%M')}"
            )

        return lines

    def generate_report(self, outage: CurrentOutage,
                        schedules: List[DaySchedule],
                        upcoming: Optional[datetime]) -> str:
        """Generate complete human-readable report"""
        lines = []

        lines.extend(self.format_header(outage))
        lines.extend(self.format_alert_status(outage, upcoming))
        lines.extend(self.format_current_status(outage))
        lines.extend(self.format_schedule(schedules))

        return "\n".join(lines)


# ============================================================================
# SCHEDULE PARSER
# ============================================================================
class DayScheduleParser:
    """Parses schedule for a single day"""

    def __init__(self, base_date: datetime, hours_map: Dict[str, str],
                 updated_at: Optional[datetime]):
        self.base_date = base_date
        self.hours_map = hours_map
        self.updated_at = updated_at
        self.periods: List[OutagePeriod] = []
        self.current_period_start: Optional[datetime] = None
        self.current_period_type: str = "yes"

    def parse(self) -> DaySchedule:
        """Parse all hours for this day"""
        sorted_hours = sorted([int(k) for k in self.hours_map.keys()])

        for hour in sorted_hours:
            status = TimeType(self.hours_map[str(hour)])
            self._process_hour(hour, status)

        self._finalize_periods()

        return DaySchedule(date=self.base_date,
                           periods=self.periods,
                           updated_at=self.updated_at)

    def _process_hour(self, hour: int, status: TimeType):
        """Process a single hour based on its status"""

        # Determine start and end of this specific hour slot
        slot_start = self._create_time(hour - 1, 0)
        slot_mid = self._create_time(hour - 1, 30)

        # Handle full hour outage (NO or MAYBE)
        if status in [TimeType.NO, TimeType.MAYBE]:
            outage_type = status.value
            if self.current_period_start is None:
                self.current_period_start = slot_start
                self.current_period_type = outage_type

        # Handle first half hour outage (FIRST or MFIRST)
        elif status in [TimeType.FIRST, TimeType.MFIRST]:
            outage_type = "maybe" if status == TimeType.MFIRST else "no"
            if self.current_period_start is None:
                self.current_period_start = slot_start
                self.current_period_type = outage_type

            self._add_period(start=self.current_period_start,
                             end=slot_mid,
                             outage_type=self.current_period_type)
            self.current_period_start = None

        # Handle second half hour outage (SECOND or MSECOND)
        elif status in [TimeType.SECOND, TimeType.MSECOND]:
            outage_type = "maybe" if status == TimeType.MSECOND else "no"
            if self.current_period_start:
                self._add_period(start=self.current_period_start,
                                 end=slot_start,
                                 outage_type=self.current_period_type)
            self.current_period_start = slot_mid
            self.current_period_type = outage_type

        # End the current outage period
        else:  # YES (Power ON)
            if self.current_period_start:
                self._add_period(start=self.current_period_start,
                                 end=slot_start,
                                 outage_type=self.current_period_type)
                self.current_period_start = None

    def _finalize_periods(self):
        """Finalize any remaining period at end of day"""
        if self.current_period_start:
            final_end = self._create_time(24, 0)
            self._add_period(start=self.current_period_start,
                             end=final_end,
                             outage_type=self.current_period_type)

    def _add_period(self, start: datetime, end: datetime, outage_type: str):
        """Add an outage period to the list"""
        self.periods.append(
            OutagePeriod(start=start, end=end, type=outage_type))

    def _create_time(self, hour: int, minute: int) -> datetime:
        """Create datetime handling special cases like hour=24"""
        if hour == 24:
            return (self.base_date + timedelta(days=1)).replace(hour=0,
                                                                minute=minute,
                                                                second=0,
                                                                microsecond=0)
        if hour == -1:
            return (self.base_date - timedelta(days=1)).replace(hour=23,
                                                                minute=minute,
                                                                second=0,
                                                                microsecond=0)
        return self.base_date.replace(hour=hour,
                                      minute=minute,
                                      second=0,
                                      microsecond=0)


class ScheduleParser:
    """Handles parsing of schedule data from DTEK API"""

    def __init__(self, data: Optional[Dict], group: Optional[str]):
        self.data = data
        self.group = group
        self.updated_at = self._parse_update_timestamp()

    def _parse_update_timestamp(self) -> Optional[datetime]:
        """Parse the update timestamp from schedule data"""
        if not self.data or not self.data.get("update"):
            return None
        try:
            # DTEK timestamp format: d.m.Y H:M
            return datetime.strptime(self.data["update"], "%d.%m.%Y %H:%M")
        except ValueError:
            return None

    def parse(self) -> List[DaySchedule]:
        """Parse schedule data for the configured group"""
        if not self.data or "data" not in self.data or not self.group:
            logger.warning(
                "Cannot parse schedule: missing data or missing group")
            return []

        group_key = f"GPV{self.group}"
        schedules = []

        for timestamp, groups_data in self.data["data"].items():
            if group_key not in groups_data:
                logger.warning(f"No data for group {group_key} on {timestamp}")
                continue
            try:
                base_date = datetime.fromtimestamp(int(timestamp))
                hours_map = groups_data[group_key]

                day_parser = DayScheduleParser(base_date, hours_map,
                                               self.updated_at)
                schedule = day_parser.parse()
                schedules.append(schedule)
            except Exception as e:
                logger.error(f"Error parsing schedule for {timestamp}: {e}")
                continue

        return schedules


# ============================================================================
# STATE MANAGEMENT
# ============================================================================
class StateManager:
    """Manages persistent state storage"""

    @staticmethod
    def load(file_path: Path) -> Dict[str, Any]:
        """Loads CurrentOutage data and schedule JSON string"""

        if not file_path.exists():
            return {
                "outage": CurrentOutage().model_dump(),
                "schedule_periods": ""
            }
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return {
                "outage": CurrentOutage().model_dump(),
                "schedule_periods": ""
            }

    @staticmethod
    def save(file_path: Path, outage: CurrentOutage, schedule_periods: str):
        """Saves composite state"""
        state_data = {
            "outage": outage.model_dump(mode='json'),
            "schedule_periods": schedule_periods
        }
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")


# ============================================================================
# MONITOR SERVICE
# ============================================================================
class MonitorService:
    """Main monitoring service orchestrator"""

    def __init__(self, ctx: MonitorContext):
        self.ctx = ctx
        self.formatter = MessageFormatter(ctx)
        self.schedules: List[DaySchedule] = []
        self.outage: Optional[CurrentOutage] = None
        self.notify_threshold: int = 25
        self.upcoming: Optional[datetime] = None

    def should_notify(self) -> Tuple[bool, str]:
        """Check if notification should be sent"""

        # Check upcoming outage
        self.upcoming = None
        for day in self.schedules:
            check = day.get_upcoming_outage(self.notify_threshold)
            if check:
                self.upcoming = check
                logger.info("Upcoming outage warning triggered")
                return True, ""

        # Load previous state for comparison
        last = StateManager.load(self.ctx.state_file)
        last_outage = CurrentOutage(**last["outage"])
        last_schedule = last["schedule_periods"]

        #Current outage status change
        if last_outage != self.outage:
            logger.info("Current outage status changed. Notifying")
            return True, "<code>ЗМІНА ВІДКЛЮЧЕННЯ</code>\n\n"

        # Check schedule change
        new_schedule = get_schedule_periods_json(self.schedules)
        if new_schedule != last_schedule:
            logger.info("Schedule plan changed. Notifying")
            return True, "<code>ЗМІНА ГРАФІКА</code>\n\n"

        return False, ""

    def run(self) -> Tuple[str, bool]:
        """Execute monitoring logic"""
        # Fetch data
        outage_data, schedule_data = DTEKMonitor().fetch(
            city=self.ctx.city, street=self.ctx.street)

        if not outage_data:
            return "❌ Помилка отримання даних про відключення", True

        # Parse data
        self.outage = parse_current_outage(outage_data, self.ctx.building)
        group = self.ctx.forced_group or self.outage.group

        # Parse schedule
        parser = ScheduleParser(schedule_data, group)
        self.schedules = parser.parse()

        # Evaluate changes
        should_notify, header = self.should_notify()

        # Save new state
        new_schedule_periods = get_schedule_periods_json(self.schedules)
        StateManager.save(file_path=self.ctx.state_file,
                          outage=self.outage,
                          schedule_periods=new_schedule_periods)

        # Generate report
        message = self.formatter.generate_report(outage=self.outage,
                                                 schedules=self.schedules,
                                                 upcoming=self.upcoming)
        return header + message, should_notify


# ============================================================================
# UTILITIES
# ============================================================================
def parse_current_outage(data: Dict, building: str) -> CurrentOutage:
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

    return CurrentOutage(sub_type=item.get("sub_type"),
                         start_date=item.get("start_date"),
                         end_date=item.get("end_date"),
                         group=group,
                         updated_at=data.get("updateTimestamp"))


def get_schedule_periods_json(schedules: List[DaySchedule]) -> str:
    """Generates JSON string for comparison, based only on date and periods"""

    data = []
    for day in schedules:
        day_data = {
            "date":
            day.date.strftime("%Y-%m-%d"),
            "periods": [{
                "start": p.start.strftime("%H:%M"),
                "end": p.end.strftime("%H:%M"),
                "type": p.type
            } for p in day.periods]
        }
        data.append(day_data)

    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def send_telegram_notification(token: str, chat_ids: Tuple[str], message: str):
    """Sends message to multiple chats using requests"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chat_id in chat_ids:
        try:
            data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            r = requests.post(url, data=data, timeout=5)
            r.raise_for_status()
            logger.info(f"Notification sent to {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send to {chat_id}: {e}")


@click.command()
@click.option('--city',
              default=lambda: os.environ.get("DTEK_CITY"),
              required=True,
              help='City for monitoring')
@click.option('--street',
              default=lambda: os.environ.get("DTEK_STREET"),
              required=True,
              help='Street for monitoring')
@click.option('--building',
              default=lambda: os.environ.get("DTEK_BUILDING"),
              required=True,
              help='Building number for monitoring')
@click.option('--forced-group',
              default=None,
              help='Force specific group for schedule parsing')
@click.option('--telegram-token',
              default=lambda: os.environ.get("BOT_TOKEN"),
              help='Telegram Bot Token')
@click.option('--chat-id', multiple=True, help='Telegram Chat ID(s)')
@click.option('--state-file',
              default=lambda: os.environ.get("DTEK_STATE_FILE"),
              help='Path to state file')
@click.option('--output',
              type=click.Choice(['text', 'html']),
              default='text',
              help="Output format")
def main(city, street, building, forced_group, telegram_token, chat_id,
         state_file, output):
    """DTEK Power Outage Monitor"""

    if not state_file:
        state_file = "last_state.json"

    # Create context
    ctx = MonitorContext(city=city,
                         street=street,
                         building=building,
                         forced_group=forced_group,
                         state_file=Path(state_file))

    # Run monitoring
    service = MonitorService(ctx)
    service.notify_threshold = 25
    message, should_notify = service.run()

    if output == 'html':
        print(message)
    else:
        console = Console()
        clean_msg = re.sub(r'<[^>]+>', '', message)
        console.print(clean_msg)

    if telegram_token and chat_id and should_notify:
        send_telegram_notification(telegram_token, chat_id, message)


if __name__ == "__main__":
    main()
