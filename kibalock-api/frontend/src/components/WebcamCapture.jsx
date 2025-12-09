import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Camera, Trash2, CheckCircle2, ScanFace, AlertCircle } from 'lucide-react'
import * as faceapi from 'face-api.js'
import toast from 'react-hot-toast'

export default function WebcamCapture({ onImagesReady, requiredImages = 3 }) {
  const [images, setImages] = useState([])
  const [isWebcamActive, setIsWebcamActive] = useState(false)
  const [faceDetected, setFaceDetected] = useState(false)
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [countdown, setCountdown] = useState(null)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const detectionIntervalRef = useRef(null)

  useEffect(() => {
    loadModels()
    return () => {
      stopWebcam()
    }
  }, [])

  useEffect(() => {
    onImagesReady(images)
  }, [images])

  const loadModels = async () => {
    try {
      const MODEL_URL = '/models'
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ])
      setModelsLoaded(true)
      toast.success('Modèles de détection faciale chargés')
    } catch (error) {
      console.error('Error loading face detection models:', error)
      toast.error('Erreur lors du chargement des modèles')
    }
  }

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
      })
      
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsWebcamActive(true)
        startFaceDetection()
        toast.success('📸 Webcam activée')
      }
    } catch (error) {
      console.error('Error accessing webcam:', error)
      toast.error('Impossible d\'accéder à la webcam')
    }
  }

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
    }
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
    }
    setIsWebcamActive(false)
    setFaceDetected(false)
  }

  const startFaceDetection = () => {
    detectionIntervalRef.current = setInterval(async () => {
      if (videoRef.current && canvasRef.current && modelsLoaded) {
        const detections = await faceapi
          .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()

        const displaySize = {
          width: videoRef.current.videoWidth,
          height: videoRef.current.videoHeight,
        }

        faceapi.matchDimensions(canvasRef.current, displaySize)

        const resizedDetections = faceapi.resizeResults(detections, displaySize)

        const ctx = canvasRef.current.getContext('2d')
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)

        if (resizedDetections.length > 0) {
          setFaceDetected(true)
          
          // Dessiner les landmarks
          faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections)
          
          // Dessiner un rectangle autour du visage
          resizedDetections.forEach(detection => {
            const box = detection.detection.box
            ctx.strokeStyle = '#a855f7'
            ctx.lineWidth = 3
            ctx.strokeRect(box.x, box.y, box.width, box.height)
          })
        } else {
          setFaceDetected(false)
        }
      }
    }, 100)
  }

  const captureImage = () => {
    if (!faceDetected) {
      toast.error('Aucun visage détecté')
      return
    }

    if (images.length >= requiredImages) {
      toast.error(`Maximum ${requiredImages} photos`)
      return
    }

    // Countdown
    let count = 3
    setCountdown(count)
    const countdownInterval = setInterval(() => {
      count--
      if (count === 0) {
        clearInterval(countdownInterval)
        setCountdown(null)
        takePicture()
      } else {
        setCountdown(count)
      }
    }, 1000)
  }

  const takePicture = () => {
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0)

    canvas.toBlob((blob) => {
      const image = {
        id: Date.now(),
        blob,
        url: URL.createObjectURL(blob),
      }
      setImages(prev => [...prev, image])
      toast.success(`✅ Photo ${images.length + 1}/${requiredImages} capturée`)
    }, 'image/jpeg', 0.95)
  }

  const deleteImage = (id) => {
    setImages(prev => prev.filter(img => img.id !== id))
    toast.success('Photo supprimée')
  }

  return (
    <div className="space-y-6">
      {/* Webcam View */}
      <div className="relative aspect-video bg-gray-900 rounded-2xl overflow-hidden border border-purple-500/30">
        {!isWebcamActive ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Camera className="w-20 h-20 text-purple-400 mb-4" />
            </motion.div>
            <button
              onClick={startWebcam}
              disabled={!modelsLoaded}
              className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-bold hover:from-purple-700 hover:to-blue-700 transition-all transform hover:scale-105 disabled:opacity-50"
            >
              {modelsLoaded ? 'Activer la webcam' : 'Chargement des modèles...'}
            </button>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full"
            />

            {/* Face Detection Indicator */}
            <motion.div
              animate={{
                opacity: faceDetected ? 1 : 0.3,
                scale: faceDetected ? 1 : 0.9,
              }}
              className="absolute top-4 left-4 bg-black/50 backdrop-blur-sm px-4 py-2 rounded-full flex items-center gap-2"
            >
              <div
                className={`w-3 h-3 rounded-full ${
                  faceDetected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
                }`}
              />
              <span className="text-white font-semibold">
                {faceDetected ? 'Visage détecté' : 'Positionnez votre visage'}
              </span>
            </motion.div>

            {/* Countdown Overlay */}
            <AnimatePresence>
              {countdown !== null && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.5 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.5 }}
                  className="absolute inset-0 flex items-center justify-center bg-black/50"
                >
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.5 }}
                    className="text-9xl font-bold text-white"
                  >
                    {countdown}
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Capture Button */}
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={captureImage}
              disabled={!faceDetected || images.length >= requiredImages}
              className="absolute bottom-4 left-1/2 transform -translate-x-1/2 w-16 h-16 bg-white rounded-full shadow-lg flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="w-14 h-14 border-4 border-purple-600 rounded-full" />
            </motion.button>

            {/* Stop Webcam Button */}
            <button
              onClick={stopWebcam}
              className="absolute top-4 right-4 px-4 py-2 bg-red-600/90 backdrop-blur-sm text-white rounded-full font-semibold hover:bg-red-700 transition-colors"
            >
              Arrêter
            </button>
          </>
        )}
      </div>

      {/* Captured Images Grid */}
      <AnimatePresence>
        {images.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-3"
          >
            <h3 className="text-white font-semibold flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-green-400" />
              Photos capturées ({images.length}/{requiredImages})
            </h3>
            <div className="grid grid-cols-3 gap-4">
              {images.map((image, idx) => (
                <motion.div
                  key={image.id}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  className="relative group"
                >
                  <img
                    src={image.url}
                    alt={`Face ${idx + 1}`}
                    className="w-full aspect-square object-cover rounded-xl border border-purple-500/30"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => deleteImage(image.id)}
                      className="absolute top-2 right-2 p-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                    <div className="absolute bottom-2 left-2 text-white font-bold">
                      Photo {idx + 1}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Instructions */}
      <div className="bg-blue-900/20 border border-blue-500/30 rounded-xl p-4">
        <h4 className="text-blue-300 font-semibold mb-2 flex items-center gap-2">
          <ScanFace className="w-5 h-5" />
          Conseils pour une bonne capture
        </h4>
        <ul className="text-gray-300 text-sm space-y-1">
          <li>• Assurez-vous d'avoir un bon éclairage</li>
          <li>• Regardez directement la caméra</li>
          <li>• Variez légèrement les angles (face, 3/4 gauche, 3/4 droite)</li>
          <li>• Gardez une expression neutre</li>
          <li>• Évitez les lunettes de soleil ou masques</li>
        </ul>
      </div>
    </div>
  )
}
