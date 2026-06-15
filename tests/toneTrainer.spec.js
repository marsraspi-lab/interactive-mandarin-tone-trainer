import { test, expect } from '@playwright/test';

test.describe('Mandarin Tone Trainer', () => {
  test('records audio and displays a score', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('h1')).toContainText('Mandarin Tone Trainer');
    await page.selectOption('#wordSelect', { index: 1 });
    await expect(page.locator('#playBtn')).toBeEnabled();
    await page.click('#recordBtn');
    await expect(page.locator('#recordBtn')).toContainText('Stop');
    await page.waitForTimeout(2000);
    await page.click('#recordBtn');
    await expect(page.locator('#score')).toContainText(/%/);
  });

  test('shows status message after recording', async ({ page }) => {
    await page.goto('/');
    await page.selectOption('#wordSelect', { index: 1 });
    await page.click('#recordBtn');
    await page.waitForTimeout(2000);
    await page.click('#recordBtn');
    await expect(page.locator('#status')).toContainText(/complete|pitch/i);
  });

  test('play button enables when word selected', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#playBtn')).toBeDisabled();
    await page.selectOption('#wordSelect', { index: 1 });
    await expect(page.locator('#playBtn')).toBeEnabled();
  });

  test('canvas is present and has correct dimensions', async ({ page }) => {
    await page.goto('/');
    const canvas = page.locator('#canvas');
    await expect(canvas).toBeVisible();
    await expect(canvas).toHaveAttribute('width', '800');
    await expect(canvas).toHaveAttribute('height', '400');
  });
});
