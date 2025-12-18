import React from 'react'
import { motion } from 'framer-motion'

export default function Training() {
  return (
    <div className="min-h-screen p-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto"
      >
        <div className="backdrop-blur-xl bg-gray-900/40 border border-purple-500/30 rounded-3xl p-8 shadow-2xl">
          <h1 className="text-4xl font-bold text-white mb-4">Entraînement</h1>
          <p className="text-gray-300 mb-8">
            Améliorez votre modèle biométrique en temps réel
          </p>
          <p className="text-gray-400">
            Composant Training en cours de développement...
          </p>
        </div>
      </motion.div>
    </div>
  )
}
