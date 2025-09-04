-- Create postgres user with password
CREATE USER postgres WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE music_vectors TO postgres;
ALTER USER postgres CREATEDB;
