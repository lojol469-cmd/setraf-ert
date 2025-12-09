import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import axios from 'axios'

// Auto-detect API URL from multiple sources
const getApiUrl = () => {
  // 1. From Vite env (auto-detected or .env)
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL
  }
  
  // 2. From auto-detection in vite.config.js
  if (import.meta.env.VITE_API_URL_AUTO) {
    return import.meta.env.VITE_API_URL_AUTO
  }
  
  // 3. Try to detect from current location
  const protocol = window.location.protocol
  const hostname = window.location.hostname
  
  // If running on non-localhost, use the same host
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
    return `${protocol}//${hostname}:8000`
  }
  
  // 4. Fallback to localhost
  return 'http://localhost:8000'
}

const API_URL = getApiUrl()

console.log('🔗 KibaLock API URL:', API_URL)

// Test API connectivity
const testApiConnection = async () => {
  try {
    const response = await axios.get(`${API_URL}/health`, { timeout: 5000 })
    console.log('✅ API connectée:', response.data)
    return true
  } catch (error) {
    console.warn('⚠️ API non accessible:', error.message)
    return false
  }
}

// Auto-test connection on load
testApiConnection()

export const useAuthStore = create(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,
      token: null,
      loading: false,
      error: null,
      apiUrl: API_URL,

      // Register user avec biométrie
      register: async (username, email, voiceSamples, faceImages) => {
        set({ loading: true, error: null })
        try {
          const formData = new FormData()
          formData.append('username', username)
          formData.append('email', email)
          
          voiceSamples.forEach((sample, idx) => {
            formData.append(`voice_${idx}`, sample)
          })
          
          faceImages.forEach((image, idx) => {
            formData.append(`face_${idx}`, image)
          })

          const response = await axios.post(`${API_URL}/api/auth/register`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 60000, // 60s for biometric processing
          })

          set({ 
            user: response.data.user,
            token: response.data.token,
            isAuthenticated: true,
            loading: false
          })

          return { success: true, data: response.data }
        } catch (error) {
          const errorMessage = error.response?.data?.message || error.message || 'Registration failed'
          set({ 
            error: errorMessage,
            loading: false 
          })
          return { success: false, error: errorMessage }
        }
      },

      // Login avec biométrie
      login: async (voiceSample, faceImage) => {
        set({ loading: true, error: null })
        try {
          const formData = new FormData()
          formData.append('voice', voiceSample)
          formData.append('face', faceImage)

          const response = await axios.post(`${API_URL}/api/auth/login`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 30000, // 30s for verification
          })

          set({ 
            user: response.data.user,
            token: response.data.token,
            isAuthenticated: true,
            loading: false
          })

          return { success: true, data: response.data }
        } catch (error) {
          const errorMessage = error.response?.data?.message || error.message || 'Login failed'
          set({ 
            error: errorMessage,
            loading: false 
          })
          return { success: false, error: errorMessage }
        }
      },

      // Logout
      logout: () => {
        set({ 
          user: null,
          token: null,
          isAuthenticated: false,
          error: null 
        })
      },

      // Clear error
      clearError: () => set({ error: null }),
      
      // Get API URL
      getApiUrl: () => API_URL,
    }),
    {
      name: 'kibalock-auth',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)
