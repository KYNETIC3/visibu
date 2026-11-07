import { TrackingCheck } from './screens/TrackingCheck';

export function App() {
  return (
    <>
      <header className="site-header">
        <h1>Visibu</h1>
        <p>Coffee counter conversations between ASL signers and baristas.</p>
      </header>
      <p className="interpreter-notice" role="note">
        Visibu is a student prototype. It recognises a small set of signs and does not replace a
        qualified ASL interpreter.
      </p>
      <main>
        <TrackingCheck />
      </main>
    </>
  );
}
