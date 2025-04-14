
export default function Stocks() {
    const assets = [{
        id: 1,
        symbol: 'MSFT',
        name: 'Microsoft Corporation',
        price: 300.00,
        priceChange: 0.5,
        priceChangePercent: 0.2,
        volume: 1000000,
        marketCap: 2000000000,
        high: 310.00,
        low: 290.00,
        open: 295.00,
        close: 300.00,
        lastUpdated: '2023-10-01T12:00:00Z',
        exchange: 'NASDAQ',
        currency: 'USD',
        description: 'Microsoft Corporation is an American multinational technology company with headquarters in Redmond, Washington.',
        logo: 'https://logo.clearbit.com/microsoft.com',
        website: 'https://www.microsoft.com',
        sector: 'Technology',
        industry: 'Software - Infrastructure',
        country: 'United States',
        employees: 181000,
        founded: 1975,
        ceo: 'Satya Nadella',
        timestamp: 'a minute ago',
      },
      {
        id: 2,
        symbol: 'AAPL',
        name: 'Apple Inc.',
        price: 150.00,
        priceChange: -0.5,
        priceChangePercent: -0.3,
        volume: 2000000,
        marketCap: 2500000000,
        high: 160.00,
        low: 140.00,
        open: 145.00,
        close: 150.00,
        lastUpdated: '2023-10-01T12:00:00Z',
        exchange: 'NASDAQ',
        currency: 'USD',
        description: 'Apple Inc. is an American multinational technology company headquartered in Cupertino, California.',
        logo: 'https://logo.clearbit.com/apple.com',
        website: 'https://www.apple.com',
        sector: 'Technology',
        industry: 'Consumer Electronics',
        country: 'United States',
        employees: 147000,
        founded: 1976,
        ceo: 'Tim Cook',
        timestamp: 'a minute ago',
      }]

      return (
        <>
        {assets.map(asset => {
            return (
              <p key={asset.id}>
                <b>{asset.name}</b> &mdash; {asset.symbol}
                <br />
              </p>
            );
          })}
        </>);
}
