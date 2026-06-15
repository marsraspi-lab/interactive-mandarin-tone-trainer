import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.js',
  use: {
    baseURL: 'http://localhost:3000',
    launchOptions: {
      args: [
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        '--use-file-for-fake-audio-capture=tests/fixtures/e2e/user_tone2_test.wav',
      ],
    },
  },
  webServer: {
    command: 'npx serve src -p 3000',
    port: 3000,
    reuseExistingServer: true,
  },
});
