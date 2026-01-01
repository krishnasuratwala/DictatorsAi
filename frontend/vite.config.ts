
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, (process as any).cwd(), '');
  return {
    plugins: [react()],
    define: {
      'process.env.API_KEY': JSON.stringify(env.VITE_API_KEY),
      'process.env.VITE_LLM_URL': JSON.stringify(env.VITE_LLM_URL),
    },
    server: {
      proxy: {
        // Proxy Chat requests
        '/chat': {
          target: 'http://127.0.0.1:5000',
          changeOrigin: true,
          secure: false,
        },
        // Proxy Database/Auth requests
        '/api': {
          target: 'http://127.0.0.1:5000',
          changeOrigin: true,
          secure: false,
        }
      }
    }
  }
})
