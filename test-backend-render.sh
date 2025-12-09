#!/bin/bash

# Test de connexion au backend Render

echo "════════════════════════════════════════════════════"
echo "   🧪 TEST CONNEXION BACKEND RENDER"
echo "════════════════════════════════════════════════════"
echo ""

BACKEND_URL="https://setraf-auth.onrender.com"

# Test 1: Health Check
echo "1️⃣  Test Health Check..."
HEALTH=$(curl -s "$BACKEND_URL/api/health")
if [ $? -eq 0 ]; then
    echo "✅ Health check OK"
    echo "   Response: $(echo $HEALTH | head -c 100)..."
else
    echo "❌ Health check FAIL"
fi
echo ""

# Test 2: Info serveur
echo "2️⃣  Test Info Serveur..."
INFO=$(curl -s "$BACKEND_URL/")
if [ $? -eq 0 ]; then
    echo "✅ Info serveur OK"
    echo "   $(echo $INFO | grep -o '"message":"[^"]*"' | cut -d'"' -f4)"
else
    echo "❌ Info serveur FAIL"
fi
echo ""

# Test 3: Endpoint auth (devrait retourner 404 car c'est un POST)
echo "3️⃣  Test Endpoint Auth..."
AUTH=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/api/auth/login")
if [ "$AUTH" = "404" ] || [ "$AUTH" = "405" ]; then
    echo "✅ Endpoint auth existe (code: $AUTH)"
else
    echo "⚠️  Code inattendu: $AUTH"
fi
echo ""

# Test 4: Response time
echo "4️⃣  Test Response Time..."
TIME=$(curl -o /dev/null -s -w "%{time_total}\n" "$BACKEND_URL/api/health")
echo "⏱️  Temps de réponse: ${TIME}s"
echo ""

# Test 5: SSL Certificate
echo "5️⃣  Test SSL Certificate..."
SSL=$(curl -sI "$BACKEND_URL" | grep -i "HTTP")
if echo "$SSL" | grep -q "200"; then
    echo "✅ SSL/HTTPS OK"
else
    echo "⚠️  SSL Response: $SSL"
fi
echo ""

echo "════════════════════════════════════════════════════"
echo "   ✅ Tests terminés"
echo "════════════════════════════════════════════════════"
echo ""
echo "🔗 Backend URL: $BACKEND_URL"
echo "📖 Documentation: $BACKEND_URL/api/docs"
echo ""
