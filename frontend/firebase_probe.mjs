import { initializeApp } from 'firebase/app';
import { getDatabase, get, ref } from 'firebase/database';

const firebaseConfig = {
  apiKey: 'AIzaSyD6NhJ3PnXegvwlHE-t9HS2W2YRkzSaQU8',
  authDomain: 'ai-powered-smart-health.firebaseapp.com',
  databaseURL: 'https://ai-powered-smart-health-default-rtdb.asia-southeast1.firebasedatabase.app',
  projectId: 'ai-powered-smart-health',
  storageBucket: 'ai-powered-smart-health.firebasestorage.app',
  messagingSenderId: '524395909710',
  appId: '1:524395909710:web:04abdcc719cba1c6c5a91f'
};

const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

const paths = ['hospital/patient1', 'hospital/patient1/patient1', 'hospital'];
for (const p of paths) {
  try {
    const snap = await get(ref(db, p));
    console.log('\nPATH:', p);
    console.log('EXISTS:', snap.exists());
    console.log('VALUE:', JSON.stringify(snap.val()));
  } catch (e) {
    console.log('\nPATH:', p);
    console.log('ERROR:', e.message);
  }
}
