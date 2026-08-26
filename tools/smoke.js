// Headless smoke test for demake game.html — drives the locally installed
// Edge via puppeteer-core (no browser download required).
//
// Setup (one-time, outside the repo so it stays untracked):
//   mkdir $env:TEMP\opencode\smoke; cd $env:TEMP\opencode\smoke
//   npm init -y; npm i puppeteer-core
//   copy this file there as smoke.js
//
// Usage:
//   node smoke.js                      # default fallback manifest (wave_shooter)
//   node smoke.js top_down_action_rpg  # any of the 5 template ids
//
// Verifies: page boots, no JS errors, correct scene active, header text,
// and writes a screenshot to %TEMP%\opencode\smoke\shot_<template>.png
const puppeteer = require('puppeteer-core');

const EDGE_PATHS = [
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
];

async function main() {
  const template = process.argv[2] || '';
  const url = `file:///C:/Users/j/Projects/demake-engine/frontend/game.html` +
              (template ? `?template=${template}` : '');
  const exePath = EDGE_PATHS.find(p => require('fs').existsSync(p));
  if (!exePath) { console.error('NO EDGE FOUND'); process.exit(2); }

  const browser = await puppeteer.launch({
    executablePath: exePath,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--window-size=520,420'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 500, height: 400 });

  const errors = [];
  page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));
  page.on('console', m => {
    if (m.type() === 'error' || m.type() === 'warning')
      errors.push(m.type().toUpperCase() + ': ' + m.text());
    else console.log('[console]', m.text());
  });

  console.log('URL:', url);
  await page.goto(url, { waitUntil: 'load' });
  await new Promise(r => setTimeout(r, 3500));

  const header = await page.$eval('#header', el => el.textContent)
    .catch(() => '(no header)');
  const overlayVisible = await page.$eval('#loading-overlay',
    el => getComputedStyle(el).display !== 'none').catch(() => null);
  const hasCanvas = await page.$('#game-container canvas') !== null;

  // Probe Phaser scene state if possible
  const sceneInfo = await page.evaluate(() => {
    try {
      const game = window._game;
      if (!game) return 'no phaser game';
      return game.scene.scenes.map(s => {
        const active = s.scene.isActive() ? '*' : '';
        return s.constructor.name + active;
      }).join(', ');
    } catch (e) { return 'probe failed: ' + e.message; }
  });

  console.log('HEADER:', header);
  console.log('OVERLAY VISIBLE:', overlayVisible);
  console.log('CANVAS PRESENT:', hasCanvas);
  console.log('SCENES:', sceneInfo);
  console.log(errors.length ? 'ERRORS:\n' + errors.join('\n') : 'NO CONSOLE ERRORS');

  const shot = process.env.TEMP + '\\opencode\\smoke\\shot_' +
    (template || 'default') + '.png';
  await page.screenshot({ path: shot });
  console.log('SCREENSHOT:', shot);

  await browser.close();
}
main().catch(e => { console.error('FATAL:', e.message); process.exit(1); });
