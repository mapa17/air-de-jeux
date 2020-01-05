const electron = require('electron');
const url = require('url');
const path = require('path');
const fs = require('fs');

// Some helper functions
const isMac = process.platform === 'darwin'
const isDev = process.env.NODE_ENV !== 'production'

// Import stuff from electron?
const {app, BrowserWindow, Menu, ipcMain} = electron;

let MainWindow;
let AddItemWindow;

app.on('ready', function()
{
    console.info("Resource folder is: " + process.resourcesPath);
    console.info("__dir folder is: " + __dirname);

    var basename = 'some.txt'
    var p1 = path.join(__dirname, basename);
    var p2 = path.join(process.resourcesPath, basename);

    /*
    try {
        var text1 = fs.readFileSync(p1);
        console.info("Reading form " + p1 + " ...\n" + text1)
    }
    catch (e) {
        console.log('Loading from path1 failed!')
        console.log(e);
    }
    
    try{
        var text2 = fs.readFileSync(p2);
        console.info("Reading form " + p2 + " ...\n" + text2)
    } catch (e){
        console.log('Loading from path2 failed!')
        console.log(e);
    }
    */

    // Create new window
    mainWindow = new BrowserWindow();
    // Load html in window
    //mainWindow.loadURL(`file://${__dirname}/mainWindow.html`);
    mainWindow.loadURL('file://'+p2);
    //mainWindow.webContents.openDevTools();
    // Quit app when closed
    mainWindow.on('closed', function(){
      app.quit();
    });
  });