console.log(model)
function multiply(a, b) {
    var aNumRows = a.length, aNumCols = a[0].length,
        bNumRows = b.length, bNumCols = b[0].length,
        m = new Array(aNumRows);  // initialize array of rows
    for (var r = 0; r < aNumRows; ++r) {
        m[r] = new Array(bNumCols); // initialize the current row
        for (var c = 0; c < bNumCols; ++c) {
            m[r][c] = 0;             // initialize the current cell
            for (var i = 0; i < aNumCols; ++i) {
                m[r][c] += a[r][i] * b[i][c];
            }
        }
    }
    return m;
}

function add(a, b) {
    return [a[0].map((x, i) => x + b[0][i])];
}

function sigmoid(input) {
    return [input[0].map((x) => 1 / (1 + Math.exp(-x)))];
}

function relu(input) {
    return [input[0].map((x) => Math.max(0, x))];
}

function softmax(input) {
    var sum = input[0].reduce((a, b) => a + Math.exp(b), 0);
    return [input[0].map((x) => Math.exp(x) / sum)];
}

function forward(model, input) {
    var output = input;
    for (var i = 0; i < model.length; i++) {
        weighted = multiply(output, model[i].layer.weights);
        biased  = add(weighted, model[i].layer.biases);
        switch (model[i].activation) {
            case 'Sigmoid':
                output = sigmoid(biased);
                break;
            case 'ReLU':
                output = relu(biased);
                break;
            case 'Softmax':
                output = softmax(biased);
                break;
            default:
                output = biased;
        }
    }
    return output;
}

function predict(input) {
    var output = forward(model, [centerInput(input).flat()]);
    let labelMap = {
        0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74, 20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84, 30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101, 40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116,
    };
    let predictions = Object.entries(output[0]).sort((a,b)=>{
        return b[1] - a[1]
    });
    //let letters = predictions.map(x => String.fromCharCode(labelMap[x[0]]))
    let predContainer = document.querySelector("#predictions")
    predContainer.innerHTML = ""
    predictions.slice(0,10).forEach((prediction) => {
        let li = document.createElement('li')
        li.classList.add("horizontal")
        li.innerHTML = `<span>${String.fromCharCode(labelMap[prediction[0]])}: ${Math.round(prediction[1] * 1000) / 10}% </span> <div class="bar"><div style="width: ${prediction[1] * 100}%"></div></div>`
        predContainer.appendChild(li)
    })
}

function centerInput(input) {
    //get bounding box
    let left = 27, right = 0, bottom, top
    input.forEach((row, index) => {
        let l, r
        let includes = false
        row.forEach((point,index) => {
            if(point == 1){includes = true}
            if(l == undefined && point == 1) {l = index}
            if(point == 1) {r = index}
        })
        if(l!=undefined, r!=undefined){
            left = Math.min(l, left)
            right = Math.max(r, right)
        }
        if(includes){
            if(top == undefined) {top = index}
            bottom = index
        }
    })
    centerX = Math.floor((left + right)/2)
    centerY = Math.floor((top + bottom)/2)
    shiftX = 13 - centerX
    shiftY = 13 - centerY

    centered = Array(28).fill(0).map(() => Array(28).fill(0))
    centered.forEach((row,y)=>{
        row.forEach((point,x)=>{
            let shifted = input[y - shiftY]?.[x - shiftX]
            if(shifted){
                centered[y][x] = shifted
            }
        })
    })

    return centered


}

//init canvas

let input = Array(28).fill(0).map(() => Array(28).fill(0))

let canvas= document.querySelector("#canvas")
let ctx = canvas.getContext('2d')
ctx.imageSmoothingEnabled = false

function updateCanvas(input){

    let canvasData = ctx.getImageData(0,0,canvas.width, canvas.height)
    input.forEach((row, y) => {
        row.forEach((point, x) => {
            let index = (x + y * canvas.width) * 4
            canvasData.data[index + 0] = point * 255
            canvasData.data[index + 1] = point * 255
            canvasData.data[index + 2] = point * 255
            canvasData.data[index + 3] = 255
        })
    })
    ctx.putImageData(canvasData, 0,0)
}

updateCanvas(input)

let mouseIsDown = false
let mouseX = 0
let mouseY = 0
let prevMouseX = 0
let prevMouseY = 0

canvas.addEventListener("mousemove", (e)=>{
    rect = canvas.getBoundingClientRect()
    mouseX = Math.floor((e.clientX - rect.left)/rect.width * canvas.width)
    mouseY = Math.floor((e.clientY - rect.top)/rect.height * canvas.height)
    drawCanvas()
})

canvas.addEventListener("touchmove",(e)=>{
    e.preventDefault()
    rect = canvas.getBoundingClientRect()
    mouseX = Math.floor((e.touches[0].clientX - rect.left)/rect.width * canvas.width)
    mouseY = Math.floor((e.touches[0].clientY - rect.top)/rect.height * canvas.height)
    drawCanvas()
})

canvas.addEventListener("mousedown", ()=>{mouseIsDown = true;drawCanvas() })
window.addEventListener("mouseup", ()=>{mouseIsDown = false;drawCanvas() })

canvas.addEventListener("touchstart", ()=>{mouseIsDown = true;drawCanvas() })
window.addEventListener("touchend", ()=>{mouseIsDown = false;drawCanvas() })

function drawCanvas(){
    if(prevMouseX == mouseX && prevMouseY == mouseY){return}
    if(mouseIsDown){
        prevMouseX = mouseX
        prevMouseY = mouseY
        input[mouseY][mouseX] = 1
        updateCanvas(input)
        predict(input)
    }
}

function resetCanvas(){
    input = Array(28).fill(0).map(() => Array(28).fill(0))
    updateCanvas(input)
    document.querySelector("#predictions").innerHTML = ""
}
