import { initializeApp } from "firebase/app";
import { getDatabase } from "firebase/database";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "AIzaSyD6NhJ3PnXegvwlHE-t9HS2W2YRkzSaQU8",
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || "ai-powered-smart-health.firebaseapp.com",
  databaseURL:
    import.meta.env.VITE_FIREBASE_DATABASE_URL ||
    "https://ai-powered-smart-health-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || "ai-powered-smart-health",
  storageBucket:
    import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || "ai-powered-smart-health.firebasestorage.app",
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || "524395909710",
  appId: import.meta.env.VITE_FIREBASE_APP_ID || "1:524395909710:web:04abdcc719cba1c6c5a91f"
};

const requiredFirebaseKeys = [
  "apiKey",
  "authDomain",
  "databaseURL",
  "projectId",
  "storageBucket",
  "messagingSenderId",
  "appId",
];

const missingKeys = requiredFirebaseKeys.filter((key) => !firebaseConfig[key]);

let database = null;
let firebaseInitError = null;

if (missingKeys.length > 0) {
  firebaseInitError =
    "Firebase config missing. Set VITE_FIREBASE_* values in frontend/.env to enable live sensor data.";
} else {
  try {
    const app = initializeApp(firebaseConfig);
    database = getDatabase(app);
  } catch (error) {
    firebaseInitError = `Firebase init failed: ${error.message}`;
  }
}

export { database, firebaseInitError };
