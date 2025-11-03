// Configuration file for the Musician Tracking frontend

export const API_BASE_URL = process.env.NODE_ENV === 'production'
  ? ''
  : 'http://localhost:8000';

export const WS_BASE_URL = process.env.NODE_ENV === 'production'
  ? `ws://${window.location.host}`
  : 'ws://localhost:8000';

export default {
  API_BASE_URL,
  WS_BASE_URL,
};
