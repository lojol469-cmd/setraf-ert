// MongoDB Initialization Script
db = db.getSiblingDB('kibalock');

// Create collections
db.createCollection('users');
db.createCollection('biometric_data');
db.createCollection('sessions');
db.createCollection('audit_logs');

// Create indexes
db.users.createIndex({ "email": 1 }, { unique: true });
db.users.createIndex({ "user_id": 1 }, { unique: true });
db.biometric_data.createIndex({ "user_id": 1 });
db.sessions.createIndex({ "token": 1 }, { unique: true });
db.sessions.createIndex({ "expires_at": 1 }, { expireAfterSeconds: 0 });
db.audit_logs.createIndex({ "timestamp": -1 });

print("✅ KibaLock database initialized");
