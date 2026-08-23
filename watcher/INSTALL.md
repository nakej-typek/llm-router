# AI Router watcher — instalace na Archi

Co to je: denní job, co stáhne z OpenRouteru aktuální nabídku modelů, vyfiltruje
**free textové** a uloží je jako `free_models.json` = zásobník, ze kterého router bere
kandidáty. Jen stdlib (urllib) → **žádný venv, žádný pip**, stačí systémový `python`.
Bez API klíče (OpenRouter models endpoint je veřejný).

`free_models.json` se zapisuje sem do složky watcheru → přes Syncthing se propíše zpět
na Windows (kde teď běží prototyp routeru). Není to secret, sync je OK.

## Instalace (user systemd timer)

```bash
# 1) unit soubory do user systemd
mkdir -p ~/.config/systemd/user
cp ~/Syncthing/archlinux/ai_router/watcher/ai-router-watcher.{service,timer} ~/.config/systemd/user/

# 2) ZKONTROLUJ cestu v .service (řádek ExecStart) — pokud Syncthing není v ~/Syncthing, uprav ji
nano ~/.config/systemd/user/ai-router-watcher.service

# 3) načti + zapni denní timer
systemctl --user daemon-reload
systemctl --user enable --now ai-router-watcher.timer

# 4) ať běží, i když nejsi přihlášený
loginctl enable-linger $USER
```

## Ověření

```bash
systemctl --user list-timers | grep watcher          # kdy poběží příště
python ~/Syncthing/archlinux/ai_router/watcher/watcher.py   # ruční test teď
cat ~/Syncthing/archlinux/ai_router/watcher/free_models.json | head
journalctl --user -u ai-router-watcher.service -n 30 # log posledního běhu
```

Hotovo — od teď se zásobník free modelů obnovuje denně sám.
