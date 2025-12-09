import React, { Component } from 'react'
import { motion } from 'framer-motion'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-purple-900 to-blue-900 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-900/40 backdrop-blur-xl border border-red-500/30 rounded-3xl p-8 max-w-2xl text-center"
          >
            <div className="text-6xl mb-4">⚠️</div>
            <h1 className="text-3xl font-bold text-white mb-4">Oups ! Une erreur est survenue</h1>
            <p className="text-gray-300 mb-6">L'application a rencontré une erreur inattendue.</p>
            <details className="text-left bg-gray-800/50 rounded-xl p-4 mb-6">
              <summary className="text-red-400 cursor-pointer mb-2">Détails de l'erreur</summary>
              <pre className="text-xs text-gray-400 overflow-auto">
                {this.state.error?.toString()}
              </pre>
            </details>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-bold hover:from-purple-700 hover:to-blue-700 transition-all"
            >
              Recharger la page
            </button>
          </motion.div>
        </div>
      )
    }

    return this.props.children
  }
}
