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
    command: 'python3 -m http.server 3000 --directory src',
    port: 3000,
    reuseExistingServer: true,
  },
});
