import React, { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Mic, Camera, Loader2, CheckCircle2, XCircle, User, Mail, Sparkles } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import toast from 'react-hot-toast'
import VoiceRecorder from '../components/VoiceRecorder'
import WebcamCapture from '../components/WebcamCapture'

export default function Register() {
  const navigate = useNavigate()
  const { register, loading } = useAuthStore()

  const [step, setStep] = useState(1) // 1: Info, 2: Voice, 3: Face, 4: Processing
  const [formData, setFormData] = useState({
    username: '',
    email: '',
  })
  const [voiceSamples, setVoiceSamples] = useState([])
  const [faceImages, setFaceImages] = useState([])

  const handleSubmit = async () => {
    if (voiceSamples.length < 3) {
      toast.error('Enregistrez au moins 3 échantillons vocaux')
      return
    }
    if (faceImages.length < 3) {
      toast.error('Capturez au moins 3 photos de votre visage')
      return
    }

    setStep(4)
    const result = await register(
      formData.username,
      formData.email,
      voiceSamples,
      faceImages
    )

    if (result.success) {
      toast.success('🎉 Inscription réussie ! Bienvenue sur KibaLock')
      setTimeout(() => navigate('/dashboard'), 2000)
    } else {
      toast.error(result.error || 'Erreur lors de l\'inscription')
      setStep(1)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-4xl"
      >
        {/* Header */}
        <div className="text-center mb-8">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            className="inline-block mb-4"
          >
            <Sparkles className="w-16 h-16 text-purple-400" />
          </motion.div>
          <h1 className="text-5xl font-bold text-white mb-4 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            KibaLock AI
          </h1>
          <p className="text-gray-300 text-lg">
            Authentification biométrique par IA - Voix & Visage
          </p>
        </div>

        {/* Progress Steps */}
        <div className="flex justify-center mb-8">
          {[1, 2, 3].map((num) => (
            <div key={num} className="flex items-center">
              <motion.div
                animate={{
                  scale: step >= num ? 1.1 : 1,
                  backgroundColor: step >= num ? '#a855f7' : '#374151',
                }}
                className="w-12 h-12 rounded-full flex items-center justify-center text-white font-bold"
              >
                {step > num ? <CheckCircle2 className="w-6 h-6" /> : num}
              </motion.div>
              {num < 3 && (
                <div
                  className={`w-24 h-1 mx-2 transition-colors ${
                    step > num ? 'bg-purple-500' : 'bg-gray-600'
                  }`}
                />
              )}
            </div>
          ))}
        </div>

        {/* Main Card */}
        <motion.div
          layout
          className="backdrop-blur-xl bg-gray-900/40 border border-purple-500/30 rounded-3xl p-8 shadow-2xl"
        >
          {/* Step 1: Information */}
          {step === 1 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
                <User className="w-8 h-8 text-purple-400" />
                Vos informations
              </h2>
              <div className="space-y-6">
                <div>
                  <label className="block text-gray-300 mb-2 font-medium">
                    Nom d'utilisateur
                  </label>
                  <input
                    type="text"
                    value={formData.username}
                    onChange={(e) =>
                      setFormData({ ...formData, username: e.target.value })
                    }
                    className="w-full px-4 py-3 bg-gray-800/50 border border-purple-500/30 rounded-xl text-white focus:outline-none focus:border-purple-500 transition-colors"
                    placeholder="john_doe"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 mb-2 font-medium">
                    Email
                  </label>
                  <input
                    type="email"
                    value={formData.email}
                    onChange={(e) =>
                      setFormData({ ...formData, email: e.target.value })
                    }
                    className="w-full px-4 py-3 bg-gray-800/50 border border-purple-500/30 rounded-xl text-white focus:outline-none focus:border-purple-500 transition-colors"
                    placeholder="john@example.com"
                  />
                </div>
                <button
                  onClick={() => {
                    if (!formData.username || !formData.email) {
                      toast.error('Veuillez remplir tous les champs')
                      return
                    }
                    setStep(2)
                  }}
                  className="w-full py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-bold text-lg hover:from-purple-700 hover:to-blue-700 transition-all transform hover:scale-105"
                >
                  Continuer
                </button>
              </div>
            </motion.div>
          )}

          {/* Step 2: Voice Samples */}
          {step === 2 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
                <Mic className="w-8 h-8 text-purple-400" />
                Enregistrement vocal
              </h2>
              <p className="text-gray-300 mb-6">
                Enregistrez 3 échantillons de votre voix (10-15 secondes chacun)
              </p>
              <VoiceRecorder
                onSamplesReady={(samples) => setVoiceSamples(samples)}
                requiredSamples={3}
              />
              <div className="flex gap-4 mt-6">
                <button
                  onClick={() => setStep(1)}
                  className="flex-1 py-3 bg-gray-700 text-white rounded-xl font-bold hover:bg-gray-600 transition-colors"
                >
                  Retour
                </button>
                <button
                  onClick={() => {
                    if (voiceSamples.length < 3) {
                      toast.error('Enregistrez 3 échantillons vocaux')
                      return
                    }
                    setStep(3)
                  }}
                  disabled={voiceSamples.length < 3}
                  className="flex-1 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-bold hover:from-purple-700 hover:to-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Continuer
                </button>
              </div>
            </motion.div>
          )}

          {/* Step 3: Face Capture */}
          {step === 3 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
                <Camera className="w-8 h-8 text-purple-400" />
                Capture faciale
              </h2>
              <p className="text-gray-300 mb-6">
                Capturez 3-5 photos de votre visage sous différents angles
              </p>
              <WebcamCapture
                onImagesReady={(images) => setFaceImages(images)}
                requiredImages={3}
              />
              <div className="flex gap-4 mt-6">
                <button
                  onClick={() => setStep(2)}
                  className="flex-1 py-3 bg-gray-700 text-white rounded-xl font-bold hover:bg-gray-600 transition-colors"
                >
                  Retour
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={faceImages.length < 3 || loading}
                  className="flex-1 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-bold hover:from-purple-700 hover:to-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Traitement...
                    </>
                  ) : (
                    'Finaliser'
                  )}
                </button>
              </div>
            </motion.div>
          )}

          {/* Step 4: Processing */}
          {step === 4 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center py-12"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="inline-block mb-6"
              >
                <Loader2 className="w-20 h-20 text-purple-400" />
              </motion.div>
              <h3 className="text-2xl font-bold text-white mb-4">
                Traitement de vos données biométriques...
              </h3>
              <p className="text-gray-400">
                Entraînement de l'IA sur votre voix et votre visage
              </p>
              <div className="mt-8 space-y-3">
                <ProcessingStep label="Extraction des embeddings vocaux" />
                <ProcessingStep label="Extraction des embeddings faciaux" />
                <ProcessingStep label="Entraînement du modèle personnalisé" />
                <ProcessingStep label="Synchronisation avec LifeModo API" />
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Footer Link */}
        <p className="text-center text-gray-400 mt-6">
          Déjà inscrit ?{' '}
          <button
            onClick={() => navigate('/login')}
            className="text-purple-400 hover:text-purple-300 font-semibold transition-colors"
          >
            Se connecter
          </button>
        </p>
      </motion.div>
    </div>
  )
}

function ProcessingStep({ label }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex items-center gap-3 text-gray-300"
    >
      <Loader2 className="w-4 h-4 animate-spin text-purple-400" />
      {label}
    </motion.div>
  )
}
