import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/generate': {
        target: 'http://127.0.0.1:5000', // The backend server (your Flask API)
        changeOrigin: true, // This ensures the origin of the request is modified to match the target server
      },
    },
  },
})
