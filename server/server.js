const express = require('express');
const path = require('path');
const { exec } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

// JSON 요청의 본문을 파싱하도록 설정
app.use(express.json());

// 정적 파일 제공 (HTML, CSS, JS 파일)
app.use(express.static(path.join(__dirname, 'public')));

// 기본 GET 요청에 대해 index.html 파일 제공
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// POST 요청에 대해 watermark.exe 실행
app.post('/executeWatermark', (req, res) => {
    let imgPath = req.body.imgPath;
    imgPath=path.join(__dirname,'public',imgPath);
    exec(`C:/Users/2h1/Desktop/server/public/watermark.exe ${imgPath}`, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            res.status(500).send('Internal Server Error');
            return;
        }
        console.log(`stdout: ${stdout}`);
        console.error(`stderr: ${stderr}`);
        res.send('Watermarking process completed');
    });
});

app.get('*', (req, res) => {
    const requestedFile = path.join(__dirname, 'public', req.path + '.html');
    res.sendFile(requestedFile, (err) => {
        if (err) {
            res.status(404).sendFile(path.join(__dirname, 'public', '404.html'));
        }
    });
});
// 모든 GET 요청에 대해 404.html 제공 (없는 페이지 요청 시)
app.use((req, res) => {
    res.status(404).sendFile(path.join(__dirname, 'public', '404.html'));
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});