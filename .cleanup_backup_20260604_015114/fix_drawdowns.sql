-- 1. Crear la tabla de drawdowns si no existe
CREATE TABLE IF NOT EXISTS drawdowns (
    id SERIAL PRIMARY KEY,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    drawdown_pct FLOAT NOT NULL,
    period VARCHAR(20) NOT NULL
);

-- 2. Hacer dueño al usuario 'trading' y dar permisos
ALTER TABLE drawdowns OWNER TO trading;
GRANT ALL PRIVILEGES ON TABLE drawdowns TO trading;
GRANT ALL PRIVILEGES ON TABLE drawdowns TO PUBLIC;

-- 3. Si se creó una secuencia para el ID, también le damos permisos
DO $$ 
DECLARE 
    r RECORD;
BEGIN 
    FOR r IN (SELECT sequence_name FROM information_schema.sequences WHERE sequence_name LIKE 'drawdowns%') LOOP 
        EXECUTE 'ALTER SEQUENCE ' || r.sequence_name || ' OWNER TO trading;'; 
    END LOOP; 
END $$;
