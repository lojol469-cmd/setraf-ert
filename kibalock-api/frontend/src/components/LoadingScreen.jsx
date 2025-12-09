import React from 'react'
import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'

export default function LoadingScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-purple-900 to-blue-900">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-center"
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="inline-block mb-6"
        >
          <Loader2 className="w-20 h-20 text-purple-400" />
        </motion.div>
        <h2 className="text-3xl font-bold text-white mb-4">Chargement...</h2>
        <p className="text-gray-400">Préparation de l'interface KibaLock</p>
      </motion.div>
    </div>
  )
}
