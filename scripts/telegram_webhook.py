#!/usr/bin/env python3
"""Manage the bot's Telegram webhook."""

import ipaddress
import os
import re
from urllib.parse import urlsplit

import click
import requests
from rich import print_json

API_URL = "https://api.telegram.org/bot"
ALLOWED_UPDATES = ["message"]
SECRET_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,256}\Z")


class Webhook:
    def __init__(self, token):
        self.token = token

    def get_webhook_info(self):
        return self._request("getWebhookInfo")

    def delete_webhook(self, drop_pending_updates=False):
        return self._request(
            "deleteWebhook", {"drop_pending_updates": drop_pending_updates}
        )

    def set_webhook(
        self, hook_url, secret_token, max_connections=None, ip_address=None
    ):
        data = {
            "url": hook_url,
            "allowed_updates": ALLOWED_UPDATES,
            "secret_token": secret_token,
        }
        if max_connections is not None:
            data["max_connections"] = max_connections
        if ip_address:
            data["ip_address"] = ip_address
        return self._request("setWebhook", data)

    def _request(self, method_name, data=None):
        if self.token is None:
            self.token = click.prompt("Bot token", hide_input=True)
        try:
            response = requests.request(
                "POST" if data is not None else "GET",
                API_URL + self.token + "/" + method_name,
                json=data,
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            raise click.ClickException("Telegram API request failed") from exc
        except ValueError as exc:
            raise click.ClickException("Telegram returned invalid JSON") from exc
        if not isinstance(result, dict) or not result.get("ok"):
            description = (
                result.get("description", "Unknown Telegram API error")
                if isinstance(result, dict)
                else "Invalid Telegram API response"
            )
            raise click.ClickException(description)
        return result


def validate_url(url):
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme == "https" and parsed.hostname
    except ValueError:
        valid = False
    if not valid:
        raise click.BadParameter("must be an HTTPS URL with a host", param_hint="--url")
    return url


def validate_ip(ip_address):
    if ip_address:
        try:
            ipaddress.ip_address(ip_address)
        except ValueError as exc:
            raise click.BadParameter(
                "must be a valid IP address", param_hint="--ip-address"
            ) from exc
    return ip_address


def get_secret(secret_token=None, interactive=False):
    if secret_token is None:
        secret_token = os.getenv("TELEGRAM_WEBHOOK_SECRET")
        if (
            interactive
            and secret_token
            and not click.confirm(
                "Use TELEGRAM_WEBHOOK_SECRET from the environment?", default=True
            )
        ):
            secret_token = None
    if secret_token is None:
        secret_token = click.prompt("Telegram webhook secret token", hide_input=True)
    if not SECRET_PATTERN.fullmatch(secret_token):
        raise click.BadParameter(
            "must be 1-256 characters using letters, digits, _ or -",
            param_hint="--secret-token",
        )
    return secret_token


def current_webhook(wh):
    info = wh.get_webhook_info()["result"]
    if not info.get("url"):
        raise click.ClickException("No webhook is set. Use create instead.")
    return info


def show_settings(url, ip_address, max_connections):
    click.echo(f"URL: {url}")
    click.echo(f"Pinned IP: {ip_address or 'none (use DNS)'}")
    click.echo(f"Allowed updates: {', '.join(ALLOWED_UPDATES)}")
    limit = max_connections if max_connections is not None else "Telegram default"
    click.echo(f"Max connections: {limit}")
    click.echo("Secret: provided (hidden)")


def create_webhook(wh, url, ip_address, max_connections, secret_token, confirm=False):
    if wh.get_webhook_info()["result"].get("url"):
        raise click.ClickException("A webhook is already set. Use update instead.")
    if url is None:
        url = click.prompt("Webhook URL")
    if confirm:
        ip_address = click.prompt(
            "Pinned IP (optional)", default="", show_default=False
        )
        max_connections = click.prompt(
            "Max connections", default=40, type=click.IntRange(1, 100)
        )
    url = validate_url(url)
    ip_address = validate_ip(ip_address)
    secret_token = get_secret(secret_token, interactive=confirm)
    if confirm:
        show_settings(url, ip_address, max_connections)
        if not click.confirm("Create this webhook?", default=False):
            return
    print_json(data=wh.set_webhook(url, secret_token, max_connections, ip_address))


def update_webhook(
    wh,
    url=None,
    ip_address=None,
    clear_ip_address=False,
    secret_token=None,
    interactive=False,
):
    if ip_address and clear_ip_address:
        raise click.UsageError(
            "--ip-address and --clear-ip-address cannot be used together"
        )
    info = current_webhook(wh)
    if interactive:
        url = click.prompt("Webhook URL", default=info["url"])
        click.echo(f"Current delivery IP: {info.get('ip_address', 'unknown')}")
        ip_address = click.prompt(
            "IP to pin (leave blank to use DNS)", default="", show_default=False
        )
    url = validate_url(url or info["url"])
    ip_address = validate_ip(None if clear_ip_address else ip_address)
    secret_token = get_secret(secret_token, interactive=interactive)
    max_connections = info.get("max_connections")
    if interactive:
        show_settings(url, ip_address, max_connections)
        if not click.confirm("Update this webhook?", default=False):
            return
    print_json(data=wh.set_webhook(url, secret_token, max_connections, ip_address))


def delete_webhook(wh, drop_pending_updates=False):
    info = current_webhook(wh)
    click.echo(f"URL: {info['url']}")
    click.echo(f"Pending updates: {info.get('pending_update_count', 'unknown')}")
    click.echo(f"Discard pending updates: {'yes' if drop_pending_updates else 'no'}")
    if click.confirm("Delete this webhook?", default=False):
        print_json(data=wh.delete_webhook(drop_pending_updates))


def interactive_menu(wh):
    if wh.token is None:
        wh.token = click.prompt("Bot token", hide_input=True)
    while True:
        click.echo("\n1. Info  2. Create  3. Update  4. Delete  q. Quit")
        choice = click.prompt(
            "Choose an action",
            type=click.Choice(["1", "2", "3", "4", "q"], case_sensitive=False),
        )
        if choice == "q":
            return
        try:
            if choice == "1":
                print_json(data=wh.get_webhook_info())
            elif choice == "2":
                create_webhook(wh, None, None, None, None, confirm=True)
            elif choice == "3":
                update_webhook(wh, interactive=True)
            else:
                drop = click.confirm("Discard pending updates?", default=False)
                delete_webhook(wh, drop)
        except click.ClickException as exc:
            exc.show()


@click.group(invoke_without_command=True)
@click.option("--bot-token", envvar="BOT_TOKEN", help="Bot token (prompts if unset)")
@click.pass_context
def cli(ctx, bot_token):
    ctx.obj = Webhook(bot_token)
    if ctx.invoked_subcommand is None:
        interactive_menu(ctx.obj)


@cli.command()
@click.pass_obj
def info(wh):
    """Show the current webhook settings."""
    print_json(data=wh.get_webhook_info())


@cli.command()
@click.option(
    "--drop-pending-updates", is_flag=True, help="Discard undelivered updates"
)
@click.pass_obj
def delete(wh, drop_pending_updates):
    """Remove the webhook."""
    delete_webhook(wh, drop_pending_updates)


@cli.command()
@click.option("--url", help="HTTPS URL for the new webhook")
@click.option("--ip-address", help="Pin delivery to this IP address")
@click.option(
    "--max-connections",
    "--max_connections",
    default=40,
    show_default=True,
    type=click.IntRange(1, 100),
)
@click.option(
    "--secret-token",
    "--secret_token",
    envvar="TELEGRAM_WEBHOOK_SECRET",
    hide_input=True,
)
@click.pass_obj
def create(wh, url, ip_address, max_connections, secret_token):
    """Register a new webhook."""
    create_webhook(wh, url, ip_address, max_connections, secret_token)


@cli.command()
@click.option("--url", help="New HTTPS URL (defaults to current URL)")
@click.option("--ip-address", help="Pin delivery to this IP address")
@click.option("--clear-ip-address", is_flag=True, help="Resume DNS delivery")
@click.option(
    "--secret-token",
    "--secret_token",
    envvar="TELEGRAM_WEBHOOK_SECRET",
    hide_input=True,
)
@click.pass_obj
def update(wh, url, ip_address, clear_ip_address, secret_token):
    """Change the current webhook."""
    update_webhook(wh, url, ip_address, clear_ip_address, secret_token)


if __name__ == "__main__":
    cli()
