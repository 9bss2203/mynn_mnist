const MODEL_PATH = "./satoimo_mnist_model.bin";
const WEIGHTS_LENGTH = 101770;

let ctx;
let percentageCtx;
let resultDOM;
let isDrawing;
let weights = [];

//キャンバスを初期状態に戻す
function cleanCanvas(){
	//手書き部分
	isDrawing = false;
	ctx.beginPath();
	ctx.clearRect(0,0,300,300);
	ctx.fillStyle = "#FFF";
	ctx.fillRect(0,0,28,28);
	//結果表示部分
	resultDOM.innerHTML = "";
	percentageCtx.clearRect(0,0,300,300);
}

//推論を行う
function inference(){
	if(weights.length != WEIGHTS_LENGTH) return; //モデルが読み込まれていない
	//画像からRGBを取得、値を入力層へ格納
	let imageData = ctx.getImageData(0,0,28,28).data;
	let inputData = [];
	for(let i = 0;i < 784;i++){
		let r = imageData[i*4+0];
		let g = imageData[i*4+1];
		let b = imageData[i*4+2];
		inputData.push(1-(((r+g+b)/3.0)/255));
	}
	//中間層の値の計算
	let middleData = [];
	for(let i = 0;i < 128;i++){
		let value = weights[(1+784)*i];
		for(let j = 0;j < 784;j++){
			value += inputData[j]*weights[(1+784)*i+j+1];
		}
		middleData.push(1/(1+Math.exp(-value)));
	}
	//出力層の値を計算
	let outputData = [];
	for(let i = 0;i < 10;i++){
		let value = weights[(1+784)*128+(1+128)*i];
		for(let j = 0;j < 128;j++){
			value += middleData[j]*weights[(1+784)*128+(1+128)*i+j+1];
		}
		outputData.push(1/(1+Math.exp(-value)));
	}
	//最大値を計算
	let maxValue = outputData[0];
	let maxIndex = 0;
	for(let i = 1;i < outputData.length;i++){
		if(maxValue < outputData[i]){
			maxIndex = i;
			maxValue = outputData[i];
		}
	}
	//最大の添字を表示
	resultDOM.innerHTML = maxIndex;
	//各数字の確率を表示
	percentageCtx.font = "28px sans-serif";
	percentageCtx.clearRect(0,0,300,300);
	for(let i = 0;i < 10;i++){
		let width = outputData[i]*260;
		if(i == maxIndex){
			percentageCtx.fillStyle = "#00F";
		}else{
			percentageCtx.fillStyle = "#000";
		}
		percentageCtx.fillText(i,2,30*i+25);
		percentageCtx.fillRect(30,30*i+5,width,20);
	}
}

//モデルを読み込み
function loadModel(){
	let req = new XMLHttpRequest();
	req.addEventListener("load",()=>{
		let arrayBuffer = req.response;
		if(arrayBuffer){
			let view = new DataView(arrayBuffer);
			for(let i = 0;i < view.byteLength/4;i++){
				weights.push(view.getFloat32(i*4,true)); //重みはfloat型で格納されている
			}
		}
		if(weights.length != WEIGHTS_LENGTH){ //重みの長さが違う
			alert("モデルの読み込みに失敗しました。");
		}
	});
	req.addEventListener("error",()=>{
		alert("モデルの読み込みに失敗しました。");
	});
	req.open("GET",MODEL_PATH);
	req.responseType = "arraybuffer";
	req.send();
}


//書き始め
function handleStart(x,y){
	isDrawing = true;
	ctx.beginPath();
	ctx.moveTo(x,y);
}

//書いている途中
function handleMove(x,y){
	if(isDrawing){
		ctx.lineTo(x,y);
		ctx.stroke();
		inference();
	}	
}

//書き終わり
function handleEnd(x,y){
	ctx.lineTo(x,y);
	ctx.stroke();
	isDrawing = false;
	inference();
}


window.addEventListener("load",()=>{
	//変数を設定
	let canvas = document.getElementsByTagName("canvas")[0];
	ctx = canvas.getContext("2d");
	resultDOM = document.getElementById("result");
	percentageCtx = document.getElementById("percentage").getContext("2d");

	//キャンバスをリセット
	cleanCanvas();

	//マウスイベントを登録
	canvas.addEventListener("mousedown",(e)=>{
		let x = e.offsetX/300*28;
		let y = e.offsetY/300*28;
		handleStart(x,y);
	});
	canvas.addEventListener("mousemove",(e)=>{
		let x = e.offsetX/300*28;
		let y = e.offsetY/300*28;
		handleMove(x,y);
	});
	canvas.addEventListener("mouseup",(e)=>{
		let x = e.offsetX/300*28;
		let y = e.offsetY/300*28;
		handleEnd(x,y);
	});
	//タッチイベントを登録
	canvas.addEventListener("touchstart",(e)=>{
		let x = (e.changedTouches[0].pageX-canvas.getBoundingClientRect().left-window.pageXOffset)/300*28;
		let y = (e.changedTouches[0].pageY-canvas.getBoundingClientRect().top -window.pageYOffset)/300*28;
		handleStart(x,y);
	});
	canvas.addEventListener("touchmove",(e)=>{
		let x = (e.changedTouches[0].pageX-canvas.getBoundingClientRect().left-window.pageXOffset)/300*28;
		let y = (e.changedTouches[0].pageY-canvas.getBoundingClientRect().top -window.pageYOffset)/300*28;
		handleMove(x,y);
	});
	canvas.addEventListener("touchend",(e)=>{
		let x = (e.changedTouches[0].pageX-canvas.getBoundingClientRect().left-window.pageXOffset)/300*28;
		let y = (e.changedTouches[0].pageY-canvas.getBoundingClientRect().top -window.pageYOffset)/300*28;
		handleEnd(x,y);
	});

	//モデルの読み込み
	loadModel();
});
