import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import os from 'os'

// Auto-detect local IP
function getLocalIP() {
  const interfaces = os.networkInterfaces()
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      // Skip internal and non-IPv4 addresses
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address
      }
    }
  }
  return 'localhost'
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const localIP = getLocalIP()
  
  // Auto-detect API URLs
  const apiUrl = env.VITE_API_URL || `http://${localIP}:8000`
  const backendUrl = env.VITE_BACKEND_URL || `http://${localIP}:8505`
  
  console.log('🔍 Auto-detection réseau:')
  console.log(`  • IP locale: ${localIP}`)
  console.log(`  • API LifeModo: ${apiUrl}`)
  console.log(`  • Backend KibaLock: ${backendUrl}`)
  
  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: parseInt(env.VITE_PORT || '3000'),
      host: '0.0.0.0', // Listen on all interfaces
      proxy: {
        '/api': {
          target: apiUrl,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '/api'),
        },
        '/socket.io': {
          target: apiUrl,
          changeOrigin: true,
          ws: true,
        },
      },
    },
    define: {
      // Expose env variables to client
      'import.meta.env.VITE_LOCAL_IP': JSON.stringify(localIP),
      'import.meta.env.VITE_API_URL_AUTO': JSON.stringify(apiUrl),
      'import.meta.env.VITE_BACKEND_URL_AUTO': JSON.stringify(backendUrl),
    },
  }
})
