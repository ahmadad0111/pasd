// Persistent network connection that will be used to transmit real-time data
//var socket = io();

var socket = io(window.location.origin);


var config;

var tutorial_instructions = () => [
    `
    <p>Mechanic: <b>Delivery</b></p>
    <p>Your goal here is to cook and deliver soups in order to earn reward. Notice how your partner is busily churning out soups</p>
    <p>See if you can copy his actions in order to cook and deliver the appropriate soup</p>
    <p><b>Note</b>: only recipes in the <b>All Orders</b> field will earn reward. Thus, you must cook a soup with <b>exactly</b> 3 onions</p>
    <p><b>You will advance only when you have delivered the appropriate soup</b></p>
    <p>Good luck!</p>
    <br></br>
    `,
    `
    <p>Mechanic: <b>Delivery</b></p>
    <p>Your goal here is to cook and deliver soups in order to earn reward. Notice how your partner is busily churning out soups</p>
    <p>See if you can copy his actions in order to cook and deliver the appropriate soup</p>
    <p><b>Note</b>: only recipes in the <b>All Orders</b> field will earn reward. Thus, you must cook a soup with <b>exactly</b> 3 onions</p>
    <p><b>You will advance only when you have delivered the appropriate soup</b></p>
    <p>Good luck!</p>
    <br></br>
    `,
    `
    <p>Mechanic: <b>Delivery</b></p>
    <p>Your goal here is to cook and deliver soups in order to earn reward. Notice how your partner is busily churning out soups</p>
    <p>See if you can copy his actions in order to cook and deliver the appropriate soup</p>
    <p><b>Note</b>: only recipes in the <b>All Orders</b> field will earn reward. Thus, you must cook a soup with <b>exactly</b> 3 onions</p>
    <p><b>You will advance only when you have delivered the appropriate soup</b></p>
    <p>Good luck!</p>
    <br></br>
    `,
    `
    <p>Mechanic: <b>Delivery</b></p>
    <p>Your goal here is to cook and deliver soups in order to earn reward. Notice how your partner is busily churning out soups</p>
    <p>See if you can copy his actions in order to cook and deliver the appropriate soup</p>
    <p><b>Note</b>: only recipes in the <b>All Orders</b> field will earn reward. Thus, you must cook a soup with <b>exactly</b> 3 onions</p>
    <p><b>You will advance only when you have delivered the appropriate soup</b></p>
    <p>Good luck!</p>
    <br></br>
    `
];

var tutorial_hints = () => [
    `
    <p>
        You can move up, down, left, and right using
        the <b>arrow keys</b>, and interact with objects
        using the <b>spacebar</b>.
      </p>
      <p>
        You can interact with objects by facing them and pressing
        <b>spacebar</b>. Here are some examples:
        <ul>
          <li>You can pick up ingredients (onions or tomatoes) by facing
            the ingredient area and pressing <b>spacebar</b>.</li>
          <li>If you are holding an ingredient, are facing an empty counter,
            and press <b>spacebar</b>, you put the ingredient on the counter.</li>
          <li>If you are holding an ingredient, are facing a pot that is not full,
            and press <b>spacebar</b>, you will put the ingredient in the pot.</li>
          <li>If you are facing a pot that is non-empty, are currently holding nothing, and 
            and press <b>spacebar</b>, you will begin cooking a soup.</li>
        </ul>
      </p>
    `,
    `
    <p>
        You can move up, down, left, and right using
        the <b>arrow keys</b>, and interact with objects
        using the <b>spacebar</b>.
      </p>
      <p>
        You can interact with objects by facing them and pressing
        <b>spacebar</b>. Here are some examples:
        <ul>
          <li>You can pick up ingredients (onions or tomatoes) by facing
            the ingredient area and pressing <b>spacebar</b>.</li>
          <li>If you are holding an ingredient, are facing an empty counter,
            and press <b>spacebar</b>, you put the ingredient on the counter.</li>
          <li>If you are holding an ingredient, are facing a pot that is not full,
            and press <b>spacebar</b>, you will put the ingredient in the pot.</li>
          <li>If you are facing a pot that is non-empty, are currently holding nothing, and 
            and press <b>spacebar</b>, you will begin cooking a soup.</li>
        </ul>
      </p>    `,
    `
    <p>
        You can move up, down, left, and right using
        the <b>arrow keys</b>, and interact with objects
        using the <b>spacebar</b>.
      </p>
      <p>
        You can interact with objects by facing them and pressing
        <b>spacebar</b>. Here are some examples:
        <ul>
          <li>You can pick up ingredients (onions or tomatoes) by facing
            the ingredient area and pressing <b>spacebar</b>.</li>
          <li>If you are holding an ingredient, are facing an empty counter,
            and press <b>spacebar</b>, you put the ingredient on the counter.</li>
          <li>If you are holding an ingredient, are facing a pot that is not full,
            and press <b>spacebar</b>, you will put the ingredient in the pot.</li>
          <li>If you are facing a pot that is non-empty, are currently holding nothing, and 
            and press <b>spacebar</b>, you will begin cooking a soup.</li>
        </ul>
      </p>    `,
    `
    <p>
        You can move up, down, left, and right using
        the <b>arrow keys</b>, and interact with objects
        using the <b>spacebar</b>.
      </p>
      <p>
        You can interact with objects by facing them and pressing
        <b>spacebar</b>. Here are some examples:
        <ul>
          <li>You can pick up ingredients (onions or tomatoes) by facing
            the ingredient area and pressing <b>spacebar</b>.</li>
          <li>If you are holding an ingredient, are facing an empty counter,
            and press <b>spacebar</b>, you put the ingredient on the counter.</li>
          <li>If you are holding an ingredient, are facing a pot that is not full,
            and press <b>spacebar</b>, you will put the ingredient in the pot.</li>
          <li>If you are facing a pot that is non-empty, are currently holding nothing, and 
            and press <b>spacebar</b>, you will begin cooking a soup.</li>
        </ul>
      </p>    `
]

var curr_tutorial_phase;

// Read in game config provided by server
$(function() {
    config = JSON.parse($('#config').text());
    window.config_tut = config
    tutorial_instructions = tutorial_instructions();
    tutorial_hints = tutorial_hints();
    $('#quit').show();
});

/* * * * * * * * * * * * * * * * 
 * Button click event handlers *
 * * * * * * * * * * * * * * * */

$(function() {
    $('#try-again').click(function () {
        data = {
            "params" : config['tutorialParams'],
            "game_name" : "tutorial"
        };
        socket.emit("join", data);
        $('try-again').attr("disable", true);
    });
});

$(function() {
    $('#show-hint').click(function() {
        let text = $(this).text();
        let new_text = text === "Show Hint" ? "Hide Hint" : "Show Hint";
        $('#hint-wrapper').toggle();
        $(this).text(new_text);
    });
});

$(function() {
    $('#quit').click(function() {
        socket.emit("leave", {});
        $('quit').attr("disable", true);
        window.location.href = "./";
    });
});

$(function() {
    $('#finish').click(function() {
        $('#finish').attr("disable", true);
         window.location.href = "./";
    });
});

function showStartModal(message = null) {
  const modal = $('#start-modal');
  if (message) {
    modal.find('.start-modal-content').text(message);

  }
  modal.fadeIn(300);

  setTimeout(() => {
    modal.fadeOut(300);
  }, 1000);
}

function showTutOverModal(message = null) {
  const modal = $('#tutover-modal');
  if (message) {
    modal.find('.tutover-modal-content').text(message);

  }
  modal.fadeIn(300);

  setTimeout(() => {
    modal.fadeOut(300);
  }, 1000);
}

/* * * * * * * * * * * * * 
 * Socket event handlers *
 * * * * * * * * * * * * */

socket.on('creation_failed', function(data) {
    // Tell user what went wrong
    let err = data['error']
    $("#overcooked").empty();
    $('#overcooked').append(`<h4>Sorry, tutorial creation code failed with error: ${JSON.stringify(err)}</>`);
    $('#try-again').show();
    $('#try-again').attr("disabled", false);
});

socket.on('start_game', function(data) {
    curr_tutorial_phase = 1;
    graphics_config = {
        container_id : "overcooked",
        start_info : data.start_info
    };
    $("#overcooked").empty();
    $('#game-over').hide();
    $('#try-again').hide();
    $('#try-again').attr('disabled', true)
    $('#hint-wrapper').hide();
    $('#show-hint').text('Show Hint');
    $('#game-title').text(`Tutorial in Progress, Phase ${curr_tutorial_phase}/${tutorial_instructions.length}`);
    $('#game-title').show();
    $('#tutorial-instructions').html(tutorial_instructions[curr_tutorial_phase-1]);
    $('#instructions-wrapper').show();
    $('#hint').append(tutorial_hints[curr_tutorial_phase]);
    showStartModal(message=`Game Started! Phase ${curr_tutorial_phase}`);
    enable_key_listener();
    graphics_start(graphics_config);
});

socket.on('reset_game', function(data) {
    curr_tutorial_phase++;
    graphics_end();
    disable_key_listener();
    $("#overcooked").empty();
    $('#tutorial-instructions').empty();
    $('#hint').empty();
    $("#tutorial-instructions").html(tutorial_instructions[curr_tutorial_phase-1]);
    $("#hint").append(tutorial_hints[curr_tutorial_phase]);
    if (curr_tutorial_phase <= tutorial_instructions.length) {
      $('#game-title').text(`Tutorial in Progress, Phase ${curr_tutorial_phase}/${tutorial_instructions.length}`);
      showStartModal(message=`Game Started! Phase ${curr_tutorial_phase}`);
    } else {
      $('#game-title').hide();
    }
    console.log(data)
    let button_pressed = $('#show-hint').text() === 'Hide Hint';
    if (button_pressed) {
        $('#show-hint').click();
    }
    graphics_config = {
        container_id : "overcooked",
        start_info : data.state
    };
    graphics_start(graphics_config);
    enable_key_listener();
});

socket.on('state_pong', function(data) {
    // Draw state update
    drawState(data['state']);
});

socket.on('end_game', function(data) {
    // Hide game data and display game-over html
    graphics_end();
    disable_key_listener();
    $('#game-title').hide();
    $('#instructions-wrapper').hide();
    $('#hint-wrapper').hide();
    $('#show-hint').hide();
    $('#game-over').show();
    $('#quit').hide();
    
    if (data.status === 'inactive') {
        // Game ended unexpectedly
        $('#error-exit').show();
        // Propogate game stats to parent window 
        window.top.postMessage({ name : "error" }, "*");
    } else {
        // Propogate game stats to parent window 
        window.top.postMessage({ name : "tutorial-done" }, "*");
    }

    $('#finish').show();
    console.log(data)
    // Extract layout
    const layoutName = data.data.trajectory[0].layout_name;

    // Determine human player ID
    let humanPlayerId = '';
    const t = data.data.trajectory[0];

    if (t.player_0_is_human) {
      humanPlayerId = t.player_0_id;
    } else if (t.player_1_is_human) {
      humanPlayerId = t.player_1_id;
    }
    window.surveyParams = {
      player_id: humanPlayerId,
      uid: data.data.uid,
      pre_game: false, // disabled for HRL
      pre_game_link: config.questionnaire_links.pre_game
    }
    let surveyURL = `${config.questionnaire_links.demographic}?player_Id=${humanPlayerId}&uid=${data.data.uid}&layout=${layoutName}`;
    showQualtricsSurvey(surveyURL)
});

/* * * * * * * * * * * * * 
 * Qualtrics handlers    *
 * * * * * * * * * * * * */
// JavaScript logic to control Qualtrics iframe modal
window.addEventListener('message', function (event) {
  console.log(event.data)
  if (event.data === 'qualtricsSubmitted') {
    console.log("Survey completed. Showing Close button.");
    // $('#close-qualtrics').removeAttr('hidden');
     // 1. Close the modal and clear the iframe
    const modal = document.getElementById('qualtrics-modal');
    const iframe = document.getElementById('qualtrics-frame');
    modal.style.display = 'none';
    iframe.src = '';  // Optional: clear iframe
    setTimeout(() => {
      if (window.surveyParams["pre_game"]) {
        // open next suevry
        console.log(" show pre-game")
        let surveyURL = `${window.surveyParams.pre_game_link}?player_Id=${window.surveyParams.player_id}&uid=${window.surveyParams.uid}`;
        showQualtricsSurvey(surveyURL)
        window.surveyParams.isLastSurvey = true
      }
      delete window.surveyParams.player_id
      delete window.surveyParams.uid
      delete window.surveyParams['pre_game']
      delete window.surveyParams['pre_game_link']
    }, 500); // Delay to make the transition smooth

    if(window.surveyParams.isLastSurvey){
      showTutOverModal();
      delete window.surveyParams['isLastSurvey']

    }
  }
});

function showQualtricsSurvey(url) {
  const modal = document.getElementById('qualtrics-modal');
  const iframe = document.getElementById('qualtrics-frame');
  iframe.src = url;  // dynamically set URL based on participant/layout
  modal.style.display = 'block';
}

function closeQualtricsSurvey() {
  const modal = document.getElementById('qualtrics-modal');
  const iframe = document.getElementById('qualtrics-frame');
  iframe.src = "";  // clear iframe to reset survey
  modal.style.display = 'none';
  if (window.surveyParams["pre_game"]) {
    // open next suevry
    let surveyURL = `${window.surveyParams.pre_game_link}?player_Id=${window.surveyParams.player_id}&uid=${window.surveyParams.uid}`;
    showQualtricsSurvey(surveyURL)
  }

  delete window.surveyParams.player_id
  delete window.surveyParams.uid
  delete window.surveyParams['pre_game']
  delete window.surveyParams['pre_game_link']
  
}

document.addEventListener('DOMContentLoaded', () => {
  const closeBtn = document.getElementById('close-qualtrics');
  if (closeBtn) {
    closeBtn.addEventListener('click', closeQualtricsSurvey);
  }
});



/* * * * * * * * * * * * * * 
 * Game Key Event Listener *
 * * * * * * * * * * * * * */

function enable_key_listener() {
    $(document).on('keydown', function(e) {
        let action = 'STAY'
        switch (e.which) {
            case 37: // left
                action = 'LEFT';
                break;

            case 38: // up
                action = 'UP';
                break;

            case 39: // right
                action = 'RIGHT';
                break;

            case 40: // down
                action = 'DOWN';
                break;

            case 32: //space
                action = 'SPACE';
                break;

            default: // exit this handler for other keys
                return; 
        }
        e.preventDefault();
        socket.emit('action', { 'action' : action });
    });
};

function disable_key_listener() {
    $(document).off('keydown');
};

/* * * * * * * * * * * * 
 * Game Initialization *
 * * * * * * * * * * * */

socket.on("connect", function() {
    // Config for this specific game
    config = JSON.parse($('#config').text());
    let data = {
        "params" : config['tutorialParams'],
        "game_name" : "tutorial"
    };

    // create (or join if it exists) new game
    socket.emit("join", data);
});


/* * * * * * * * * * *
 * Utility Functions *
 * * * * * * * * * * */

var arrToJSON = function(arr) {
    let retval = {}
    for (let i = 0; i < arr.length; i++) {
        elem = arr[i];
        key = elem['name'];
        value = elem['value'];
        retval[key] = value;
    }
    return retval;
};