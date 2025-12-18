import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Mic, Square, Trash2, CheckCircle2, Waves } from 'lucide-react'
import RecordRTC from 'recordrtc'
import toast from 'react-hot-toast'

export default function VoiceRecorder({ onSamplesReady, requiredSamples = 3 }) {
  const [isRecording, setIsRecording] = useState(false)
  const [samples, setSamples] = useState([])
  const [currentDuration, setCurrentDuration] = useState(0)
  const [audioLevel, setAudioLevel] = useState(0)

  const recorderRef = useRef(null)
  const streamRef = useRef(null)
  const analyserRef = useRef(null)
  const animationFrameRef = useRef(null)
  const timerRef = useRef(null)

  useEffect(() => {
    return () => {
      stopRecording()
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  useEffect(() => {
    onSamplesReady(samples)
  }, [samples])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        } 
      })
      streamRef.current = stream

      // Audio analyser pour visualisation
      const audioContext = new AudioContext()
      const analyser = audioContext.createAnalyser()
      const source = audioContext.createMediaStreamSource(stream)
      source.connect(analyser)
      analyser.fftSize = 256
      analyserRef.current = analyser

      // Démarrer la visualisation
      animateAudioLevel()

      // Recorder
      recorderRef.current = new RecordRTC(stream, {
        type: 'audio',
        mimeType: 'audio/webm',
        sampleRate: 44100,
        desiredSampRate: 16000,
        recorderType: RecordRTC.StereoAudioRecorder,
        numberOfAudioChannels: 1,
      })

      recorderRef.current.startRecording()
      setIsRecording(true)
      setCurrentDuration(0)

      // Timer
      timerRef.current = setInterval(() => {
        setCurrentDuration(prev => {
          if (prev >= 15) {
            stopRecording()
            return 15
          }
          return prev + 0.1
        })
      }, 100)

      toast.success('🎤 Enregistrement démarré')
    } catch (error) {
      console.error('Error starting recording:', error)
      toast.error('Impossible d\'accéder au microphone')
    }
  }

  const stopRecording = () => {
    if (!recorderRef.current) return

    recorderRef.current.stopRecording(() => {
      const blob = recorderRef.current.getBlob()
      
      if (currentDuration < 3) {
        toast.error('Enregistrement trop court (minimum 3 secondes)')
      } else if (samples.length >= requiredSamples) {
        toast.error(`Maximum ${requiredSamples} échantillons`)
      } else {
        const sample = {
          id: Date.now(),
          blob,
          duration: currentDuration.toFixed(1),
          url: URL.createObjectURL(blob),
        }
        setSamples(prev => [...prev, sample])
        toast.success(`✅ Échantillon ${samples.length + 1}/${requiredSamples} enregistré`)
      }

      // Cleanup
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }

      setIsRecording(false)
      setAudioLevel(0)
      setCurrentDuration(0)
    })
  }

  const animateAudioLevel = () => {
    if (!analyserRef.current) return

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(dataArray)

    const average = dataArray.reduce((a, b) => a + b) / dataArray.length
    setAudioLevel(average / 255)

    animationFrameRef.current = requestAnimationFrame(animateAudioLevel)
  }

  const deleteSample = (id) => {
    setSamples(prev => prev.filter(s => s.id !== id))
    toast.success('Échantillon supprimé')
  }

  return (
    <div className="space-y-6">
      {/* Visualizer */}
      <div className="relative h-48 bg-gradient-to-br from-purple-900/30 to-blue-900/30 rounded-2xl overflow-hidden border border-purple-500/30">
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            animate={{
              scale: isRecording ? [1, 1.2, 1] : 1,
              opacity: isRecording ? [0.5, 1, 0.5] : 0.3,
            }}
            transition={{
              duration: 2,
              repeat: isRecording ? Infinity : 0,
              ease: "easeInOut",
            }}
            className="w-32 h-32 rounded-full bg-purple-500/20 backdrop-blur-sm"
          />
          <motion.div
            animate={{
              scale: 1 + audioLevel * 0.5,
              opacity: 0.5 + audioLevel * 0.5,
            }}
            className="absolute w-24 h-24 rounded-full bg-purple-500 flex items-center justify-center"
          >
            <Mic className="w-12 h-12 text-white" />
          </motion.div>
        </div>

        {/* Waveform */}
        {isRecording && (
          <div className="absolute bottom-0 left-0 right-0 h-16 flex items-end justify-center gap-1 px-4">
            {[...Array(40)].map((_, i) => (
              <motion.div
                key={i}
                animate={{
                  height: `${Math.random() * audioLevel * 100}%`,
                }}
                transition={{
                  duration: 0.1,
                  repeat: Infinity,
                  repeatType: "reverse",
                }}
                className="flex-1 bg-gradient-to-t from-purple-500 to-blue-500 rounded-t-full"
                style={{ minHeight: '4px' }}
              />
            ))}
          </div>
        )}

        {/* Timer */}
        {isRecording && (
          <div className="absolute top-4 right-4 bg-red-500/90 backdrop-blur-sm px-4 py-2 rounded-full text-white font-mono font-bold">
            {currentDuration.toFixed(1)}s
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex gap-4">
        {!isRecording ? (
          <button
            onClick={startRecording}
            disabled={samples.length >= requiredSamples}
            className="flex-1 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-bold text-lg hover:from-purple-700 hover:to-blue-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Mic className="w-6 h-6" />
            Enregistrer ({samples.length}/{requiredSamples})
          </button>
        ) : (
          <button
            onClick={stopRecording}
            className="flex-1 py-4 bg-red-600 text-white rounded-xl font-bold text-lg hover:bg-red-700 transition-all transform hover:scale-105 flex items-center justify-center gap-2"
          >
            <Square className="w-6 h-6 fill-current" />
            Arrêter
          </button>
        )}
      </div>

      {/* Samples List */}
      <AnimatePresence>
        {samples.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-3"
          >
            <h3 className="text-white font-semibold flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-green-400" />
              Échantillons enregistrés
            </h3>
            {samples.map((sample, idx) => (
              <motion.div
                key={sample.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="bg-gray-800/50 border border-purple-500/30 rounded-xl p-4 flex items-center gap-4"
              >
                <div className="w-12 h-12 rounded-full bg-purple-600 flex items-center justify-center text-white font-bold">
                  {idx + 1}
                </div>
                <div className="flex-1">
                  <p className="text-white font-semibold">Échantillon {idx + 1}</p>
                  <p className="text-gray-400 text-sm">{sample.duration}s</p>
                </div>
                <audio src={sample.url} controls className="h-10" />
                <button
                  onClick={() => deleteSample(sample.id)}
                  className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Instructions */}
      <div className="bg-blue-900/20 border border-blue-500/30 rounded-xl p-4">
        <h4 className="text-blue-300 font-semibold mb-2 flex items-center gap-2">
          <Waves className="w-5 h-5" />
          Conseils pour un bon enregistrement
        </h4>
        <ul className="text-gray-300 text-sm space-y-1">
          <li>• Parlez clairement pendant 10-15 secondes</li>
          <li>• Évitez le bruit de fond</li>
          <li>• Variez vos phrases entre les échantillons</li>
          <li>• Restez à 20-30cm du microphone</li>
        </ul>
      </div>
    </div>
  )
}
