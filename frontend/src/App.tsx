import Container from 'react-bootstrap/Container';
import Header from './components/Header';
import Body from './components/Body';
import Stocks from './components/Stocks';

export default function App() {
  return (
    <Container fluid className="App">
      <Header />
      <Body sidebar>
        <Stocks />
      </Body>
    </Container>
  );
}
