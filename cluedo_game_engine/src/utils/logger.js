import winston from 'winston';

// Determine log level from environment variable, default to 'info'
const logLevel = process.env.LOG_LEVEL || 'info';

const logger = winston.createLogger({
  level: logLevel,
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.printf(({ level, message, timestamp }) => {
      return `${timestamp} [${level.toUpperCase()}] ${message}`;
    })
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.printf(({ level, message, timestamp }) => {
          return `${timestamp} [${level}] ${message}`;
        })
      )
    })
  ]
});

// Add a method to handle objects for debug logging
logger.debugObj = (message, obj) => {
  if (logger.level === 'debug') {
    logger.debug(`${message} ${JSON.stringify(obj, null, 2)}`);
  }
};

export { logger }; 