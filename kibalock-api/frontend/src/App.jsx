import React, { Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { motion, AnimatePresence } from 'framer-motion'

// Pages
import Dashboard from './pages/Dashboard'
import Register from './pages/Register'
import Login from './pages/Login'
import Chat from './pages/Chat'
import Training from './pages/Training'

// Components
import Scene3D from './components/Scene3D'
import LoadingScreen from './components/LoadingScreen'
import ErrorBoundary from './components/ErrorBoundary'

// Store
import { useAuthStore } from './store/authStore'

function App() {
  const { isAuthenticated, user } = useAuthStore()

  return (
    <ErrorBoundary>
      <BrowserRouter>
        <div className="relative w-full h-screen overflow-hidden bg-gradient-to-br from-gray-900 via-purple-900 to-blue-900">
          {/* Background 3D Scene */}
          <div className="absolute inset-0 z-0">
            <Scene3D />
          </div>

          {/* Main Content */}
          <div className="relative z-10 w-full h-full">
            <Suspense fallback={<LoadingScreen />}>
              <AnimatePresence mode="wait">
                <Routes>
                  <Route 
                    path="/" 
                    element={
                      isAuthenticated ? <Navigate to="/dashboard" /> : <Navigate to="/login" />
                    } 
                  />
                  <Route 
                    path="/register" 
                    element={
                      isAuthenticated ? <Navigate to="/dashboard" /> : <Register />
                    } 
                  />
                  <Route 
                    path="/login" 
                    element={
                      isAuthenticated ? <Navigate to="/dashboard" /> : <Login />
                    } 
                  />
                  <Route 
                    path="/dashboard" 
                    element={
                      isAuthenticated ? <Dashboard /> : <Navigate to="/login" />
                    } 
                  />
                  <Route 
                    path="/chat" 
                    element={
                      isAuthenticated ? <Chat /> : <Navigate to="/login" />
                    } 
                  />
                  <Route 
                    path="/training" 
                    element={
                      isAuthenticated ? <Training /> : <Navigate to="/login" />
                    } 
                  />
                </Routes>
              </AnimatePresence>
            </Suspense>
          </div>

          {/* Toast Notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 3000,
              style: {
                background: 'rgba(17, 24, 39, 0.8)',
                backdropFilter: 'blur(10px)',
                color: '#fff',
                border: '1px solid rgba(168, 85, 247, 0.3)',
                borderRadius: '12px',
                padding: '16px',
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#fff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </div>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
