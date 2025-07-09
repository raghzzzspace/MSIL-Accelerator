const { useState } = React;

function TimeSeries() {
    const [cols, setCols] = useState(null);
    const [metrics, setMetrics] = useState([]);
    const [csvText, setCsvText] = useState('');

    async function uploadFile() {
        const f = document.getElementById('tsFile');
        const file = f.files[0];
        const reader = new FileReader();

        reader.onload = async (e) => {
            const text = e.target.result;
            setCsvText(text);

            try {
                const response = await fetch('/parse/csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ csv: text })
                });

                if (!response.ok) {
                    const err = await response.text();
                    console.error("Error parsing CSV:", err);
                    alert("Failed to parse CSV.");
                    return;
                }

                const result = await response.json();
                setCols(result.columns);
            } catch (err) {
                console.error("Unexpected error:", err);
                alert("Something went wrong.");
            }
        };

        reader.readAsText(file);
    }

    async function trainModel() {
        const data = {
            csv: csvText,
            model: document.getElementById('tsModel').value,
            datetime_col: document.getElementById('tsDatetime').value,
            target_col: document.getElementById('tsTarget').value,
        };

        const response = await fetch('/train/timeseries', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        setMetrics(prev => [...prev, [
            result.metrics['MAE'],
            result.metrics['RMSE'],
            result.model
        ]]);

        plotForecast(result.dates, result.actual, result.forecast);
    }

    function plotForecast(dates, actual, forecast) {
        const residuals = actual.map((val, i) => val - forecast[i]);

        // Plot 1: Forecast vs Actual
        Plotly.newPlot('plotlyChart', [
            {
                x: dates,
                y: actual,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual'
            },
            {
                x: dates,
                y: forecast,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Forecast'
            }
        ], {
            title: 'Forecast vs Actual',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Value' }
        });

        // Plot 2: Residuals
        Plotly.newPlot('residualChart', [{
            x: dates,
            y: residuals,
            type: 'bar',
            marker: { color: 'orange' },
            name: 'Residuals'
        }], {
            title: 'Residual Plot',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Residuals (Actual - Forecast)' }
        });

        // Plot 3: Error Distribution
        Plotly.newPlot('errorDistChart', [{
            x: residuals,
            type: 'histogram',
            name: 'Error Distribution',
            marker: { color: 'teal' }
        }], {
            title: 'Error Distribution',
            xaxis: { title: 'Residual Value' },
            yaxis: { title: 'Frequency' }
        });

        // Plot 4: Actual vs Forecast scatter
        Plotly.newPlot('scatterChart', [{
            x: actual,
            y: forecast,
            mode: 'markers',
            type: 'scatter',
            name: 'Actual vs Forecast',
            marker: { color: 'purple' }
        }], {
            title: 'Actual vs Forecast Scatter',
            xaxis: { title: 'Actual' },
            yaxis: { title: 'Forecast' }
        });
    }

    return (
        <form className="timeseries-form">
            <input type="file" id="tsFile" accept=".csv" />
            <input type="button" value="Upload" onClick={uploadFile} /><br /><br />

            {cols && <>
                <label>DateTime Column: </label>
                <select id="tsDatetime">
                    {cols.map(col => <option key={col}>{col}</option>)}
                </select><br /><br />

                <label>Target Column: </label>
                <select id="tsTarget">
                    {cols.map(col => <option key={col}>{col}</option>)}
                </select><br /><br />
            </>}

            <label>Model: </label>
            <select id="tsModel">
                <option value="arima">ARIMA</option>
                <option value="prophet">Prophet</option>
                <option value="linear">Linear Regression</option>
            </select><br /><br />

            <input type="button" value="Train" onClick={trainModel} /><br /><br />

            {metrics.length > 0 && <>
                <table>
                    <thead>
                        <tr>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>Model</th>
                        </tr>
                    </thead>
                    <tbody>
                        {metrics.map((m, i) => (
                            <tr key={i}>
                                <td>{m[0]}</td>
                                <td>{m[1]}</td>
                                <td>{m[2]}</td>
                            </tr>
                        ))}
                    </tbody>
                </table><br /><br />
            </>}

            <h3>Forecast Visualizations</h3>

<div style={{
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
    gap: '30px',
    marginBottom: '50px'
}}>
    <div style={{ background: '#fff', borderRadius: '10px', padding: '10px', boxShadow: '0 2px 6px rgba(0,0,0,0.1)' }}>
        <div id="plotlyChart" style={{ width: '100%', height: '400px' }}></div>
        <p style={{ textAlign: 'center', fontWeight: 'bold' }}>Forecast vs Actual</p>
    </div>

    <div style={{ background: '#fff', borderRadius: '10px', padding: '10px', boxShadow: '0 2px 6px rgba(0,0,0,0.1)' }}>
        <div id="scatterChart" style={{ width: '100%', height: '400px' }}></div>
        <p style={{ textAlign: 'center', fontWeight: 'bold' }}>Actual vs Forecast Scatter</p>
    </div>

    <div style={{ background: '#fff', borderRadius: '10px', padding: '10px', boxShadow: '0 2px 6px rgba(0,0,0,0.1)' }}>
        <div id="residualChart" style={{ width: '100%', height: '400px' }}></div>
        <p style={{ textAlign: 'center', fontWeight: 'bold' }}>Residual Plot</p>
    </div>

    <div style={{ background: '#fff', borderRadius: '10px', padding: '10px', boxShadow: '0 2px 6px rgba(0,0,0,0.1)' }}>
        <div id="errorDistChart" style={{ width: '100%', height: '400px' }}></div>
        <p style={{ textAlign: 'center', fontWeight: 'bold' }}>Error Distribution</p>
    </div>
</div>


        </form>
    );
}
