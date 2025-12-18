import React from 'react'
import { motion } from 'framer-motion'

export default function Login() {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="backdrop-blur-xl bg-gray-900/40 border border-purple-500/30 rounded-3xl p-8 shadow-2xl max-w-2xl w-full"
      >
        <h1 className="text-4xl font-bold text-white mb-4 text-center">
          Connexion KibaLock
        </h1>
        <p className="text-gray-300 text-center mb-8">
          Utilisez votre voix et votre visage pour vous connecter
        </p>
        <p className="text-gray-400 text-center">
          Composant Login en cours de développement...
        </p>
      </motion.div>
    </div>
  )
}
