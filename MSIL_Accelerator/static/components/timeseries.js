const { useState } = React;

function TimeSeries() {
    const [cols, setCols] = useState(null);
    const [metrics, setMetrics] = useState([]);

    let colOptions = null;
    if (cols) {
        colOptions = cols.map(col => <option key={col}>{col}</option>);
    }

    async function uploadFile() {
        const data = new FormData();
        const f = document.getElementById('tsTrainFile');
        data.append('dataset', f.files[0]);

        await fetch('/upload/timeseries', {
            method: 'POST',
            body: data
        })
        .then(response => response.json())
        .then(result => setCols(result['cols']));
    }

    async function trainModel() {
        const data = {
            model: document.getElementById('tsModel').value,
            datetime_col: document.getElementById('tsDatetime').value,
            target_col: document.getElementById('tsTarget').value,
            freq: document.getElementById('tsFreq').value
        };

        await fetch('/train/timeseries', {
            method: 'POST',
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(result => {
            if (result.metrics) {
                const { MAE, RMSE } = result.metrics;
                setMetrics([...metrics, [MAE, RMSE, data.model]]);
            } else {
                alert("Training failed: " + result.error);
            }
        });
    }

    return (
        <form className='timeseries-form'>
            <input type="file" id="tsTrainFile" />
            <input type="button" value="Upload" onClick={uploadFile} /><br /><br />

            {cols && <><label>Date Column: </label>
                <select id="tsDatetime">{colOptions}</select><br /><br />
                <label>Target Column: </label>
                <select id="tsTarget">{colOptions}</select><br /><br /></>}

            <label>Model: </label>
            <select id="tsModel">
                <option value="arima">ARIMA</option>
                <option value="prophet">Prophet</option>
                <option value="linear">Linear Regression</option>
            </select><br /><br />

            <label>Frequency: </label>
            <select id="tsFreq">
                <option value="D">Daily</option>
                <option value="W">Weekly</option>
                <option value="M">Monthly</option>
            </select><br /><br />

            <input type="button" value="Train" onClick={trainModel} /><br /><br />

            {metrics.length !== 0 && <table>
                <thead>
                    <tr>
                        <th>MAE</th>
                        <th>RMSE</th>
                        <th>Model</th>
                    </tr>
                </thead>
                <tbody>
                    {metrics.map((m, idx) => (
                        <tr key={idx}>
                            <td>{m[0]}</td>
                            <td>{m[1]}</td>
                            <td>{m[2]}</td>
                        </tr>
                    ))}
                </tbody>
            </table>}
        </form>
    );
}
