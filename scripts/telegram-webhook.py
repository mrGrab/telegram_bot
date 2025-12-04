#!/usr/bin/env python3
# coding: utf-8
"""
Telegram documentation - https://core.telegram.org/bots/api
"""

import os
import click
import requests
from rich import print_json

API_URL = "https://api.telegram.org/bot"


class Webhook():

    def __init__(self, token):
        self.api_url = API_URL + token

    def get_webhook_info(self):
        url = self.api_url + "/getWebhookInfo"
        return self._send_request(url)

    def delete_webhook(self):
        url = self.api_url + "/deleteWebhook"
        data = {"drop_pending_updates": True}
        return self._send_request(url, data=data)

    def set_webhook(self, hook_url, allowed_updates, secret_token=""):
        url = self.api_url + "/setWebhook"
        data = {
            "url": hook_url,
            "allowed_updates": ["message"],
            "secret_token": secret_token
        }
        return self._send_request(url, data=data)

    def _send_request(self, url, data={}, headers={}, method="GET"):
        r = requests.request(method=method,
                             url=url,
                             headers=headers,
                             json=data)

        return r.json()


@click.group()
@click.option('--bot-token',
              default=lambda: os.getenv('BOT_TOKEN'),
              required=True,
              help='bot token')
@click.pass_context
def cli(ctx, bot_token):
    ctx.obj = Webhook(bot_token)


@click.command()
@click.pass_obj
def info(wh):
    info = wh.get_webhook_info()
    print_json(data=info)


@click.command()
@click.pass_obj
def delete(wh):
    wh_url = wh.get_webhook_info()["result"]["url"]
    if click.confirm(f"Please confirm deleting {wh_url}",
                     default=False,
                     abort=True):
        result = wh.delete_webhook()
        print_json(data=result)


@click.command()
@click.option('--url', prompt='HTTPS URL to send updates to', required=True)
@click.option('--max_connections',
              prompt='The maximum allowed number connections',
              default=40)
@click.option('--secret_token',
              prompt='A secret token to be sent in a header',
              default="")
@click.pass_obj
def create(wh, url, max_connections, secret_token):
    result = wh.set_webhook(url, max_connections, secret_token)
    print_json(data=result)


cli.add_command(info)
cli.add_command(delete)
cli.add_command(create)

if __name__ == '__main__':
    cli()
