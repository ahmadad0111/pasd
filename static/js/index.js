// Persistent network connection that will be used to transmit real-time data
// var socket = io();
var socket = io(window.location.origin);


/* * * * * * * * * * * * * * * * 
 * Button click event handlers *
 * * * * * * * * * * * * * * * */

$(function() {
    $('#create').click(function () {
        params = arrToJSON($('form').serializeArray());
        params.layouts = [params.layout]
        data = {
            "params" : params,
            "game_name" : "overcooked",
            "create_if_not_found" : false
        };
        socket.emit("create", data);
        $('#waiting').show();
        $('#join').hide();
        $('#join').attr("disabled", true);
        $('#create').hide();
        $('#create').attr("disabled", true)
        $('#create-next').hide();
        $('#create-next').attr("disabled", true)
        $("#instructions").hide();
        $('#tutorial').hide();
    });
});

$(function() {
    $('#create-next').click(function () {
        params = arrToJSON($('form').serializeArray());
        // console.log(params)
        params.layouts = [params.layout]
        params.layout = $('#current-layout').attr("current-layout")
        var agentMapping = window.config_data["layout_agent_mapping"];
        try {
            // document.getElementById("playerOne").value = agentMapping[params.layout];
            document.getElementById("playerZero").value = agentMapping[params.layout];
        }
        // catch(err) {document.getElementById("playerOne").value = window.config_data["layout_agent_mapping"][window.config_data["default_layout"]];console.log(err);}
        catch(err) {document.getElementById("playerZero").value = window.config_data["layout_agent_mapping"][window.config_data["default_layout"]];console.log(err);}

        // params.playerOne = agentMapping[params.layout]
        params.playerZero = agentMapping[params.layout]

        data = {
            "params" : params,
            "game_name" : "overcooked",
            "create_if_not_found" : false
        };
        // console.log("data ", data)
        socket.emit("create-next", data);
        $('#waiting').show();
        $('#join').hide();
        $('#join').attr("disabled", true);
        $('#create').hide();
        $('#create').attr("disabled", true)
        $('#create-next').hide();
        $('#create-next').attr("disabled", true)
        $("#instructions").hide();
        $('#tutorial').hide();
    });
});

$(function(){
    $('[data-toggle="tooltip"]').tooltip({
      delay: { "show": 100, "hide": 100 } /* Adjust delay times here */
    });
});

$(function() {
    $('#join').click(function() {
        socket.emit("join", {});
        $('#join').attr("disabled", true);
        $('#create').attr("disabled", true);
    });
});

$(function() {
    $('#leave').click(function() {
        socket.emit('leave', {});
        $('#leave').attr("disabled", true);
    });
});

// $(function() {
//     $('#current-layout').on("layoutUpdated", function() {
//         const currentLayoutValue = $(this).attr("current-layout");
//         console.log("Layout updated:", currentLayoutValue);
//         // $('#layout').val(currentLayoutValue);
//     });
// });

$(function() {
    $('#layout').change(function() {
        var layout = document.getElementById("layout").value;
        var agentMapping = window.config_data["layout_agent_mapping"];
        try {
            // document.getElementById("playerOne").value = agentMapping[layout];
            document.getElementById("playerZero").text = agentMapping[layout];
            $('#current-layout')
                .html(layout)
                .attr("current-layout", layout)
                // .trigger("layoutUpdated");

        }
        // catch(err) {document.getElementById("playerOne").value = window.config_data["layout_agent_mapping"][window.config_data["default_layout"]];console.log(err);}
        catch(err) {document.getElementById("playerZero").value = window.config_data["layout_agent_mapping"][window.config_data["default_layout"]];console.log(err);}
    });
});

window.onload = function() {
    // fetch('http://localhost:5000/get_config')
    fetch(window.location.origin + "/get_config")

    .then(response => response.json())
    .then(json => {
        console.log(json);
        window.config_data = json.config_data;
    });
}


/* * * * * * * * * * * * * 
 * Qualtrics handlers    *
 * * * * * * * * * * * * */
// JavaScript logic to control Qualtrics iframe modal
window.addEventListener('message', function (event) {
    if (event.data === 'qualtricsSubmitted') {
      console.log("Survey completed. Showing Close button.");
    //   $('#close-qualtrics').removeAttr('hidden');

    setTimeout(() => {
        // 1. Close the modal and clear the iframe
        const modal = document.getElementById('qualtrics-modal');
        const iframe = document.getElementById('qualtrics-frame');
        modal.style.display = 'none';
        iframe.src = '';  // Optional: clear iframe

        setTimeout(() => {
        if (window.surveyParams && window.surveyParams["post_game"]) {
          // open next suevry
          let surveyURL = window.surveyParams.post_game_link;
          showQualtricsSurvey(surveyURL)
          window.surveyParams.showend = true
          delete window.surveyParams['post_game']
          delete window.surveyParams['post_game_link']
        }
        
      }, 500); // Delay to make the transition smooth

    if (window.surveyParams && window.surveyParams.showend) {
        showEndingSequence();
        delete window.surveyParams['showend']
    }
      }, 500); // Delay to make the transition smooth

    }
  });

function showQualtricsSurvey(url) {
    console.log(url)
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
    setTimeout(() => {
        if (window.surveyParams && window.surveyParams["post_game"]) { // not triggered in HRL
          // open next suevry
          let surveyURL = window.surveyParams.post_game_link;
          showQualtricsSurvey(surveyURL)
          window.surveyParams.showend = true
          delete window.surveyParams['post_game']
          delete window.surveyParams['post_game_link']
        }
        
      }, 500); // Delay to make the transition smooth

    if (window.surveyParams && window.surveyParams.showend) {
        showEndingSequence();
        delete window.surveyParams['showend']
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const closeBtn = document.getElementById('close-qualtrics');
    if (closeBtn) {
      closeBtn.addEventListener('click', closeQualtricsSurvey);
    }
  });
  

/* * * * * * * * * * * * * 
 * Socket event handlers *
 * * * * * * * * * * * * */

window.intervalID = -1;
window.spectating = true;

socket.on('waiting', function(data) {
    // Show game lobby
    $('#error-exit').hide();
    $('#waiting').hide();
    $('#game-over').hide();
    $('#instructions').hide();
    $('#tutorial').hide();
    $("#overcooked").empty();
    $('#lobby').show();
    $('#join').hide();
    $('#join').attr("disabled", true)
    $('#create').hide();
    $('#create').attr("disabled", true)
    $('#leave').show();
    $('#leave').attr("disabled", false);
    if (!data.in_game) {
        // Begin pinging to join if not currently in a game
        if (window.intervalID === -1) {
            window.intervalID = setInterval(function() {
                socket.emit('join', {});
            }, 1000);
        }
    }
});

socket.on('creation_failed', function(data) {
    // Tell user what went wrong
    let err = data['error']
    $("#overcooked").empty();
    $('#lobby').hide();
    $("#instructions").show();
    $('#tutorial').show();
    $('#waiting').hide();
    $('#join').show();
    $('#join').attr("disabled", true);
    $('#create').show();
    $('#create').attr("disabled", false);
    $('#create-next').show();
    $('#create-next').attr("disabled", false);
    $('#overcooked').append(`<h4>Sorry, game creation code failed with error: ${JSON.stringify(err)}</>`);
});

socket.on('start_game', function(data) {
    // Hide game-over and lobby, show game title header
    if (window.intervalID !== -1) {
        clearInterval(window.intervalID);
        window.intervalID = -1;
    }
    graphics_config = {
        container_id : "overcooked",
        start_info : data.start_info
    };
    window.spectating = data.spectating;
    document.getElementById("experiment-order").innerHTML = "<b>Layout Order:</b> " + data.start_info["experiment_order_disp"];

    let currentLayout = data.start_info.current_layout.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    $('#current-layout')
    .html(currentLayout)
    .attr("current-layout", data.start_info.current_layout)
    // .trigger("layoutUpdated"); // Custom event
    $('#layout').val($('#current-layout').attr("current-layout"));
    // document.getElementById("playerOne").value = window.config_data["layout_agent_mapping"][data.start_info.current_layout]
    document.getElementById("playerZero").value = window.config_data["layout_agent_mapping"][data.start_info.current_layout]
    $('#xaiAgentType').val(data.start_info["xaiAgentType"]);
    $('#aiAgentType').val(data.start_info["aiAgentType"]);
    $('#error-exit').hide();
    $("#overcooked").empty();
    $('#game-over').hide();
    $('#lobby').hide();
    $('#waiting').hide();
    $('#join').hide();
    $('#join').attr("disabled", true);
    $('#create').hide();
    $('#create').attr("disabled", true)
    $("#instructions").hide();
    $('#tutorial').hide();
    $('#leave').show();
    $('#leave').attr("disabled", false)
    $('#game-title').show();
    $('#experiment-order').show();
    $('#experiment-order').attr("disabled", false)
    
    if (!window.spectating) {
        enable_key_listener();
    }
    
    graphics_start(graphics_config);
});

socket.on('reset_game', function(data) {
    graphics_end();
    if (!window.spectating) {
        disable_key_listener();
    }
    
    $("#overcooked").empty();
    $("#reset-game").show();
    setTimeout(function() {
        $("reset-game").hide();
        graphics_config = {
            container_id : "overcooked",
            start_info : data.state
        };
        if (!window.spectating) {
            enable_key_listener();
        }
        graphics_start(graphics_config);
    }, data.timeout);
});

socket.on('state_pong', function(data) {
    // Draw state update
    drawState(data['state']);
});

socket.on('end_game', function(data) {
    // Hide game data and display game-over html
    graphics_end();
    if (!window.spectating) {
        disable_key_listener();
    }
    speechSynthesis.cancel();
    $('#game-title').hide();
    $('#game-over').show();
    $("#join").show();
    $('#join').attr("disabled", true);
    if (data.data && !data.data.game_flow_on) {
        $("#create").show();
        $('#create').attr("disabled", false)
    } else {
        $("#create-next").show();
        $('#create-next').attr("disabled", false)
    }
    if (data.data && data.data.is_ending) {
        $("#create").show();
        $('#create').attr("disabled", true)

        $("#create-next").hide();
        $('#create-next').attr("disabled", true)
        // socket.emit('leave', {});
    }
    $("#instructions").show();
    $('#tutorial').show();
    $("#leave").hide();
    $('#leave').attr("disabled", true)
    
    // Game ended unexpectedly
    if (data.status === 'inactive') {
        $('#error-exit').show();
    }
    console.log(data)

    // Determine human player ID
    let humanPlayerId = '';
    const t = data.data && data.data.trajectory ? data.data.trajectory[0] : null

    if (t && t.player_0_is_human) {
      humanPlayerId = t.player_0_id;
    } else if (t && t.player_1_is_human) {
      humanPlayerId = t.player_1_id;
    }
    enable_survey = window.config_data.enable_survey
    if(enable_survey && data.data && data.data.session_ended){
        let surveyURL = `${data.data.survey_baseurl}?round_d=${data.data.round_id}&player_Id=${humanPlayerId}&uid=${data.data.uid}&session_Id=${data.data.session_id}&xai_agent=${data.data.xai_agent}&layout=${data.data.layout}`;
        showQualtricsSurvey(surveyURL)
        setTimeout(function(){}, 100)

    }
    if(enable_survey && data.data && data.data.phase_ended && data.data.survey_baseurl){
        // let surveyURL = `${data.data.survey_baseurl}?round_d=${data.data.round_id}&player_Id=${humanPlayerId}&uid=${data.data.uid}&session_Id=${data.data.session_id}&xai_agent=${data.data.xai_agent}&layout=${data.data.layout}`; //FOR ADAX
        let surveyURL = `${data.data.survey_baseurl}?round_d=${data.data.round_id}&player_Id=${humanPlayerId}&uid=${data.data.uid}&session_Id=${data.data.session_id}&ai_agent=${data.data.ai_agent}&layout=${data.data.layout}`; // FOR HRL
        showQualtricsSurvey(surveyURL)
        setTimeout(function(){}, 100)

    }
    if(enable_survey && data.data && data.data.game_ended && data.data.survey_baseurl_end){
        let endSurveyURL = `${data.data.survey_baseurl_end}?round_d=${data.data.round_id}&player_Id=${humanPlayerId}&uid=${data.data.uid}&session_Id=${data.data.session_id}&xai_agent=${data.data.xai_agent}&layout=${data.data.layout}`;
        window.surveyParams = {
            post_game: false, // disabled for HRL
            post_game_link: endSurveyURL,
            showend: true //ADDED for HRL
          }
        setTimeout(function(){}, 100)
    }
    // if (data.data && data.data.is_ending) {
    //     window.alert("Please enter UID for the next player!!")
    // }
      
});

speechSynthesis.onvoiceschanged = () => {
    const v = speechSynthesis.getVoices();
}
prev_xai_msg = ''
socket.on('xai_voice', function(data) {
    if(window.config_data["ttsEnabled"]) {
        try {
            xai_msg = data["explanation"]
            if (prev_xai_msg != xai_msg) {
                prev_xai_msg = xai_msg
                const utterance = new SpeechSynthesisUtterance(xai_msg);
                selectedVoice = speechSynthesis.getVoices().find(v => v.name === 'Google UK English Male' )
                utterance.lang= selectedVoice.lang;
                utterance.rate = 1.1;
                utterance.pitch = 1;
                utterance.voice = selectedVoice;
                speechSynthesis.cancel();
                speechSynthesis.speak(utterance); 
            }   
        } catch(e){
            console.log("Error xai, ", e)
        }
    }

})

socket.on('stop_sensors', function() {
    speechSynthesis.cancel()
})

$(document).ready(function () {
  const showModal = $('#instruction-modal').data('show-modal');
  if (showModal === true || showModal === 'true') {
    $('#instruction-modal').fadeIn();  // Show modal
  }

  $('#close-instructions').on('click', function () {
    $('#instruction-modal').fadeOut();
  });
  // Optional: if you use a button to trigger the instructions later
  $('#show-instructions').on('click', function () {
    $('#instruction-modal').fadeIn();
  });
});


function resetUID() {
    fetch("/reset_uid", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log("Reset successful:", data);
        // Optional: reload page or update UI
        $('#uid-value').css('display', 'none');
    })
    .catch(error => {
        console.error("Error resetting UID:", error);
    });
}

function showEndingSequence() {
    const modal = document.getElementById('end-modal');
    const message = document.getElementById('end-modal-message');
    const button = document.getElementById('end-modal-btn');
  
    // Step 1: Initial message
    message.textContent = "Experiment is over. Thank you for your participation. Please provide survey code: EXP2025";
    button.style.display = 'inline-block';
    modal.style.display = 'flex';
  
    button.onclick = function () {
          // Close modal or move to UID input
          modal.style.display = 'none';
          resetUID()
          // Optionally trigger UID entry or refresh
        };
        
    // // Step 2: On first OK click
    // button.onclick = function () {
  
    //   // Step 3: After delay, show OK again (or handle further logic)
    //   setTimeout(() => {
    //     message.textContent = "Please enter UID for the next player!!";

    //     button.style.display = 'inline-block';
    //     button.textContent = 'OK'; // Or "Continue"
    //     button.onclick = function () {
    //       // Close modal or move to UID input
    //       modal.style.display = 'none';
    //       resetUID()
    //       // Optionally trigger UID entry or refresh
    //     };
    //   }, 200);
    // };
  }
  
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
