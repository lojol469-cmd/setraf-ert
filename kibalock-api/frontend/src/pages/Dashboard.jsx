import React from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'

export default function Dashboard() {
  const navigate = useNavigate()
  const { user, logout } = useAuthStore()

  return (
    <div className="min-h-screen p-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto"
      >
        <div className="backdrop-blur-xl bg-gray-900/40 border border-purple-500/30 rounded-3xl p-8 shadow-2xl">
          <h1 className="text-4xl font-bold text-white mb-4">
            Bienvenue, {user?.username} !
          </h1>
          <p className="text-gray-300 mb-8">Votre dashboard KibaLock</p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-purple-600/20 p-6 rounded-xl border border-purple-500/30">
              <h3 className="text-xl font-bold text-white mb-2">Chat IA</h3>
              <p className="text-gray-400 mb-4">Discutez avec l'IA conversationnelle</p>
              <button
                onClick={() => navigate('/chat')}
                className="w-full py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
              >
                Ouvrir
              </button>
            </div>
            
            <div className="bg-blue-600/20 p-6 rounded-xl border border-blue-500/30">
              <h3 className="text-xl font-bold text-white mb-2">Entraînement</h3>
              <p className="text-gray-400 mb-4">Améliorez votre modèle biométrique</p>
              <button
                onClick={() => navigate('/training')}
                className="w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Ouvrir
              </button>
            </div>
            
            <div className="bg-green-600/20 p-6 rounded-xl border border-green-500/30">
              <h3 className="text-xl font-bold text-white mb-2">Profil</h3>
              <p className="text-gray-400 mb-4">Gérez vos informations</p>
              <button
                className="w-full py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                Ouvrir
              </button>
            </div>
          </div>
          
          <button
            onClick={logout}
            className="px-6 py-3 bg-red-600 text-white rounded-xl font-bold hover:bg-red-700 transition-colors"
          >
            Déconnexion
          </button>
        </div>
      </motion.div>
    </div>
  )
}
