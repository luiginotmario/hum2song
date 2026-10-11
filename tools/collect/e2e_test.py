"""Headless end-to-end check: python e2e_test.py <BASE_URL> (needs playwright + Chrome, /tmp/fake.wav)."""
import asyncio, sys
from playwright.async_api import async_playwright
U = sys.argv[1] + "/?PROLIFIC_PID=E2EPID&STUDY_ID=TEST_E2E&SESSION_ID=E2ESESS"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            "--use-file-for-fake-audio-capture=/tmp/fake.wav", "--autoplay-policy=no-user-gesture-required"])
        pg = await b.new_page()
        await pg.goto(U)
        await pg.click("#agree")
        await pg.click("#micTest")
        await pg.wait_for_selector("#micOk:not([disabled])", timeout=10000)
        print("mic:", await pg.text_content("#micStatus"))
        await pg.click("#micOk")
        t0 = await pg.text_content("#title"); await pg.click("#skip")
        print("skip:", t0, "->", await pg.text_content("#title"))
        for i in range(10):
            await pg.click("#record"); await pg.wait_for_timeout(9000); await pg.click("#record")
            await pg.wait_for_selector("#submit:not([disabled])", timeout=5000)
            print(i, await pg.text_content("#title"), await pg.text_content("#mode"))
            await pg.click("#submit")
            await pg.wait_for_timeout(1500)
        await pg.wait_for_url("**prolific.com/**", timeout=15000)
        print("redirect:", pg.url)
        await b.close()
asyncio.run(main())
