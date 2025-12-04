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


class MonitorContext(BaseModel):
    """Holds configuration to avoid global variables"""
    city: str
    street: str
    building: str
    forced_group: Optional[str]
    state_file: Path = Field(default=Path("last_state.json"))

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

    def __eq__(self, other) -> bool:
        """Compare two outage statuses"""
        if not isinstance(other, CurrentOutage):
            return False
        # Only compare critical fields
        return (self.start_date == other.start_date
                and self.end_date == other.end_date
                and self.sub_type == other.sub_type)


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


class StateManager:

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


def parse_current_outage(data: Dict, building: str) -> CurrentOutage:
    """Parse raw outage data"""

    if not data or not data.get("result"):
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


def _safe_time_create(base_date: datetime, hour: int, minute: int) -> datetime:
    """
    Handles hour=24 by moving to the next day at 00:00
    """
    if hour == 24:
        return (base_date + timedelta(days=1)).replace(hour=0,
                                                       minute=minute,
                                                       second=0,
                                                       microsecond=0)
    if hour == -1:
        return (base_date - timedelta(days=1)).replace(hour=23,
                                                       minute=minute,
                                                       second=0,
                                                       microsecond=0)
    return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)


def parse_schedule_for_group(data: Optional[Dict],
                             group: Optional[str]) -> List[DaySchedule]:
    """Parse raw schedule data"""

    if not data or "data" not in data or not group:
        return []

    group_key = f"GPV{group}"
    results = []

    updated_at = None
    if data.get("update"):
        try:
            updated_at = datetime.strptime(data["update"], "%d.%m.%Y %H:%M")
        except ValueError:
            pass

    # Iterate over days
    for timestamp, groups_data in data["data"].items():

        if group_key not in groups_data:
            logger.warning(f"No data for group {group_key} on {timestamp}")
            continue

        base_date = datetime.fromtimestamp(int(timestamp))
        hours_map = groups_data[group_key]
        sorted_hours = sorted([int(k) for k in hours_map.keys()])

        # Build outage periods
        periods = []
        current_period_start = None
        current_period_type = "yes"

        for h in sorted_hours:
            status = TimeType(hours_map[str(h)])

            # Determine start and end of this specific hour slot
            slot_start = _safe_time_create(base_date, h - 1, 0)
            slot_mid = _safe_time_create(base_date, h - 1, 30)

            p_type = "no"

            # Check if this hour contains an outage
            if status in [TimeType.NO, TimeType.MAYBE]:
                p_type = status.value
                if current_period_start is None:
                    current_period_start = slot_start
                    current_period_type = p_type
            elif status in [TimeType.FIRST, TimeType.MFIRST]:
                # First half is outage
                p_type = "maybe" if status == TimeType.MFIRST else "no"
                if current_period_start is None:
                    current_period_start = slot_start
                periods.append(
                    OutagePeriod(start=current_period_start,
                                 end=slot_mid,
                                 type=p_type))
                current_period_start = None

            elif status in [TimeType.SECOND, TimeType.MSECOND]:
                # Second half is outage
                p_type = "maybe" if status == TimeType.MSECOND else "no"
                if current_period_start:
                    periods.append(
                        OutagePeriod(start=current_period_start,
                                     end=slot_start,
                                     type=current_period_type))
                current_period_start = slot_mid
                current_period_type = p_type
            else:  # YES (Power ON)
                if current_period_start:
                    periods.append(
                        OutagePeriod(start=current_period_start,
                                     end=slot_start,
                                     type=current_period_type))
                    current_period_start = None

        # End of day cleanup
        if current_period_start:
            final_end = _safe_time_create(base_date, 24, 0)
            periods.append(
                OutagePeriod(start=current_period_start,
                             end=final_end,
                             type=current_period_type))

        results.append(
            DaySchedule(date=base_date, periods=periods,
                        updated_at=updated_at))

    return results


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


def generate_text_report(ctx: MonitorContext, outage: CurrentOutage,
                         schedules: List[DaySchedule],
                         upcoming: Optional[datetime]) -> str:
    """Generate human-readable report"""
    lines = []

    # Header — location
    lines.append(f"📍 <b>{ctx.city.replace('+', ' ')}, "
                 f"{ctx.street.replace('+', ' ')}, "
                 f"{ctx.building}</b>")

    if ctx.forced_group or outage.group:
        lines.append(f"  Черга: <b>{ctx.forced_group or outage.group}</b>")

    # Status
    if outage.is_active:
        lines.append("\n🔴 <b>Світло відсутнє!</b>")
    elif upcoming:
        mins = int((upcoming - datetime.now()).total_seconds() / 60)
        lines.append(f"\n⚠️ <b>Увага! Відключення через {mins} хв</b>")

    # Current status
    lines.append("\n📌 <b>Поточне відключення</b>")
    if outage.is_active:
        lines.append(f"{outage.sub_type}")
        if outage.start_date and outage.end_date:
            lines.append(
                f"🕒 {outage.start_date.strftime('%H:%M')} - {outage.end_date.strftime('%H:%M')}"
            )
    else:
        lines.append("✅ Зараз відключень немає")
    if outage.updated_at:
        lines.append(f"<i>Оновлено:</i> {outage.updated_at.strftime('%H:%M')}")

    lines.append("\n📅 <b>Графік відключень</b>")
    if schedules:
        for day in schedules:
            date_str = day.date.strftime("%a, %d.%m")
            lines.append(f"<b>{date_str}</b>:")

            if not day.periods:
                lines.append("  🔋 Без відключень")
            else:
                for p in day.periods:
                    icon = "❔" if p.type == "maybe" else "❌"
                    lines.append(
                        f"  {icon} {p.start.strftime('%H:%M')} - {p.end.strftime('%H:%M')}"
                    )
    else:
        lines.append("🔋 Без відключень")

    if schedules and schedules[0].updated_at:
        lines.append(
            f"<i>Графік оновлено:</i> {schedules[0].updated_at.strftime('%H:%M')}"
        )

    return "\n".join(lines)


def send_telegram_notification(token: str, chat_ids: Tuple[str], message: str):
    """Sends message to multiple chats using requests"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chat_id in chat_ids:
        try:
            data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            requests.post(url, data=data, timeout=5)
            logger.info(f"Notification sent to {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send to {chat_id}: {e}")


def run_monitor(city: str,
                street: str,
                building: str,
                forced_group: Optional[str] = None,
                state_file: str = "last_state.json",
                notify_threshold: int = 25) -> Tuple[str, bool]:
    """Main monitor logic"""

    ctx = MonitorContext(city=city,
                         street=street,
                         building=building,
                         forced_group=forced_group,
                         state_file=Path(state_file))

    monitor = DTEKMonitor()
    outage_data, schedule_data = monitor.fetch(city=ctx.city,
                                               street=ctx.street)
    if not outage_data:
        return "❌ Помилка отримання даних про відключення", True

    outage = parse_current_outage(outage_data, ctx.building)
    group = forced_group or outage.group
    schedules = parse_schedule_for_group(schedule_data, group)

    new_schedule_periods = get_schedule_periods_json(schedules)
    should_notify = False

    # Load previous state
    last = StateManager.load(ctx.state_file)
    last_outage = CurrentOutage(**last["outage"])
    last_schedule = last["schedule_periods"]

    # Check if schedule changed
    if last_outage != outage:
        logger.info("Current outage status changed")
        should_notify = True

    # Schedule Change
    if new_schedule_periods != last_schedule:
        logger.info("Schedule plan changed")
        should_notify = True

    # Upcoming Outage Warning
    upcoming = None
    for day in schedules:
        check = day.get_upcoming_outage(notify_threshold)
        if check:
            upcoming = check
            logger.info("Upcoming outage warning")
            should_notify = True
            break

    # Save state
    StateManager.save(ctx.state_file, outage, new_schedule_periods)

    # Generate report
    final_msg = generate_text_report(ctx, outage, schedules, upcoming)
    return final_msg, should_notify


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

    message, should_notify = run_monitor(city=city,
                                         street=street,
                                         building=building,
                                         forced_group=forced_group,
                                         state_file=state_file,
                                         notify_threshold=25)

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
