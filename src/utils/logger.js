export const logger = {
  info: (...args) => console.log(new Date().toISOString(), ...args),
  error: (...args) => console.error(new Date().toISOString(), ...args),
  debug: (...args) => process.env.DEBUG && console.log(new Date().toISOString(), ...args)
}; 