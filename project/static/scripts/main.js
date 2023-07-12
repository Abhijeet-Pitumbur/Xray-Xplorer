/* Abhijeet Pitumbur © Xray Xplorer 2023 */

// Wait for all DOM elements to be ready
$(function () {

    // Assign HTML elements to variables for easier manipulation
    const splashScreen = $('#splashScreen');
    const homePage = $('#homePage');
    const processingPage = $('#processingPage');
    const resultsPage = $('#resultsPage');
    const uploadButton = $('#uploadButton');
    const randomExampleButton = $('#randomExampleButton');
    const processingAnimation = $('#processingAnimation');
    const predictionClass = $('#predictionClass');
    const predictionWarning = $('#predictionWarning');
    const predictionBadge = $('#predictionBadge');
    const originalImage = $('#originalImage');
    const heatmapImage = $('#heatmapImage');
    const superimposedImage = $('#superimposedImage');
    const homeButton = $('#homeButton');
    const dragAndDropOverlay = $('#dragAndDropOverlay');
    const modal = $('#modal');
    const modalTitle = $('#modalTitle');
    const modalText = $('#modalText');
    const disclaimerButton = $('#disclaimerButton');
    const closeModalButton = $('#closeModalButton');

    // Initialize splash screen and window load status
    let splashShown = false;
    let windowLoaded = false;

    // Set load event listener on window
    $(window).on('load', windowLoadHandler);

    // Set timers for splash screen and window load fallback
    setTimeout(waitForSplashScreen, 2500);
    setTimeout(windowLoadHandler, 10000);

    // Handle window load event
    function windowLoadHandler() {
        windowLoaded = true;
        tryShowHomePage();
    }

    // Delay splash screen disappearance
    function waitForSplashScreen() {
        splashShown = true;
        tryShowHomePage();
    }

    // Show home page if splash screen has been fully shown and window has loaded
    function tryShowHomePage() {
        if (!splashShown || !windowLoaded || splashScreen.css('display') === 'none') {
            return;
        }
        splashScreen.animate({
            opacity: 0
        }, 250, function () {
            // Fade out splash screen
            $(this).css('display', 'none');
            $(this).html('');
            // Fade in home page
            homePage.css('display', 'flex').animate({
                opacity: 1
            }, 250);
        });
    }

    // Handle image upload for prediction
    function handleImageUpload(file) {
        if (!file) {
            return;
        }
        validateFileAndMakePrediction(file);
    }

    // Validate file format and size, then make prediction
    function validateFileAndMakePrediction(file) {
        if (validateFile(file)) {
            showProcessingPage();
            makePrediction(file);
        }
    }

    // Check for file format and size
    function validateFile(file) {
        // Only allow PNG, JPEG and WebP images
        if (file.type !== 'image/png' && file.type !== 'image/jpeg' && file.type !== 'image/webp') {
            showModal('Invalid File Format', 'This file format is invalid. Please try again with another image.');
            return false;
        }
        // Only allow images under 10 MB
        if (file.size > 10 * 1024 * 1024) {
            showModal('File Too Large', 'This file is too large to upload. Please try again with another image.');
            return false;
        }
        return true;
    }

    // Transition from home page to processing page
    function showProcessingPage() {
        homePage.animate({
            opacity: 0
        }, 250, function () {
            // Fade out home page and start processing animation
            $(this).css('display', 'none');
            startProcessingAnimation();
            // Fade in processing page
            processingPage.css('display', 'flex').animate({
                opacity: 1
            }, 250);
        });
    }

    // Send file to server for prediction
    function makePrediction(file) {
        const formData = new FormData();
        formData.append('file', file);
        // Send AJAX POST request to server with selected file as form data, with 25-second timeout
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            timeout: 25000,
            success: function (data) {
                // Handle server response
                handleServerResponse(data);
            },
            error: function () {
                // Handle any errors
                handleError();
            }
        });
    }

    // Handle server response
    function handleServerResponse(data) {
        if (validateServerResponse(data)) {
            // If server response is valid, show results
            showResults(data);
        } else {
            // If server response is invalid, throw error
            throw new Error();
        }
    }

    // Ensure that response includes all necessary data
    function validateServerResponse(data) {
        return data.prediction_class && data.original_image && data.heatmap_image && data.superimposed_image && data.prediction_probability && data.chest_xray_probability;
    }

    // Transition from processing page to results page
    function showResults(data) {
        processingPage.animate({
            opacity: 0
        }, 250, function () {
            // Fade out processing page and stop processing animation
            $(this).css('display', 'none');
            stopProcessingAnimation();
            // Populate results and fade in results page
            populateResults(data);
            resultsPage.css('display', 'flex').animate({
                opacity: 1
            }, 250);
        });
    }

    // Populate results of prediction
    function populateResults(data) {
        // Populate prediction class
        predictionClass.html(data.prediction_class);
        // Populate prediction images
        originalImage.attr('src', data.original_image);
        heatmapImage.attr('src', data.heatmap_image);
        superimposedImage.attr('src', data.superimposed_image);
        // Clear prediction icons
        predictionWarning.css('display', 'none');
        predictionBadge.css('display', 'none');
        if (data.chest_xray_probability <= 50) {
            // Show warning if image is not chest X-ray
            predictionWarning.css('display', 'inline-block');
            showModal('No Chest X-Ray Detected', 'The uploaded image may not be a chest X-ray. The accuracy of classification and visualizations could be adversely affected, potentially leading to nonsensical visual output. Please upload a clear chest X-ray for optimal results.');
        } else if (data.prediction_probability >= 48) {
            // Show badge if prediction probability is high
            predictionBadge.css('display', 'inline-block');
        }
    }

    // Handle random example request
    function tryRandomExample() {
        showProcessingPage();
        fetchRandomExample();
    }

    // Fetch random example
    function fetchRandomExample() {
        // Send AJAX POST request to server to fetch random example as BLOB, with 25-second timeout
        $.ajax({
            url: '/random-example',
            type: 'POST',
            timeout: 25000,
            xhrFields: {
                responseType: 'blob'
            },
            success: function (data) {
                // Handle image upload of fetched example
                handleImageUpload(new Blob([data], {
                    type: 'image/png'
                }));
            },
            error: function () {
                // Handle any errors
                handleError();
            }
        });
    }

    // Go to home page and show modal with error message
    function handleError() {
        goToHomePage();
        showModal('Error', 'An error occurred while processing the request. Please try again.');
    }

    // Fade in modal with given title and text
    function showModal(title, text) {
        modalTitle.html(title);
        modalText.html(text);
        modal.css('display', 'flex').animate({
            opacity: 1
        }, 250);
    }

    // Fade out modal
    function closeModal() {
        modal.animate({
            opacity: 0
        }, 250, function () {
            $(this).css('display', 'none');
            modalTitle.html('');
            modalText.html('');
        });
    }

    // Transition from processing page or results page to home page
    function goToHomePage() {
        if (processingPage.css('display') !== 'none') {
            processingPage.animate({
                opacity: 0
            }, 250, function () {
                // Fade out processing page and stop processing animation
                $(this).css('display', 'none');
                stopProcessingAnimation();
                // Fade in home page
                homePage.css('display', 'flex').animate({
                    opacity: 1
                }, 250);
            });
            return;
        }
        resultsPage.animate({
            opacity: 0
        }, 250, function () {
            // Fade out results page and clear any previous results
            $(this).css('display', 'none');
            clearResults();
            // Fade in home page
            homePage.css('display', 'flex').animate({
                opacity: 1
            }, 250);
        });
    }

    // Clear any previous results
    function clearResults() {
        predictionClass.html('');
        originalImage.attr('src', '');
        heatmapImage.attr('src', '');
        superimposedImage.attr('src', '');
        predictionWarning.css('display', 'none');
        predictionBadge.css('display', 'none');
    }

    // Handle file input change event on upload button
    uploadButton.on('change', function () {
        handleImageUpload(this.files[0]);
        uploadButton.val('');
    });

    // Handle click event on random example button
    randomExampleButton.on('click', function () {
        tryRandomExample();
    });

    // Handle click event on home button
    homeButton.on('click', function () {
        goToHomePage();
    });

    // Handle click event on disclaimer button
    disclaimerButton.on('click', function () {
        showModal('Disclaimer', 'This web application, "Xray Xplorer", is for educational and research purposes only, and should not be used as a direct replacement for professional medical advice or diagnosis.<br><br>Xray Xplorer uses artificial intelligence (AI) to analyze chest X-ray images, searching for potential indications of COVID-19 abnormalities, among other classifications. While this tool is technologically sophisticated, its interpretations are not infallible, and should not serve as the sole basis for any medical decision-making. This application does not replace the expertise and clinical judgement of healthcare professionals. Always consult your physician or other qualified health provider with any questions you may have regarding a medical condition.<br><br>By using Xray Xplorer, you acknowledge that the developer, Abhijeet Pitumbur, the educational institution, and all parties involved in creating or maintaining this tool are not responsible for any outcomes resulting from its use.<br><br>Abhijeet Pitumbur © Xray Xplorer 2023 · All Rights Reserved.');
    });

    // Handle click event on close modal button
    closeModalButton.on('click', function () {
        closeModal();
    });

    // Initialize Lottie animation object with specified configuration
    const lottieProcessingAnimation = lottie.loadAnimation({
        container: processingAnimation.get(0),
        renderer: 'svg',
        loop: true,
        autoplay: false,
        path: processingAnimation.attr('data-processing-animation-path')
    });

    // Start Lottie processing animation
    function startProcessingAnimation() {
        lottieProcessingAnimation.play();
    }

    // stop Lottie processing animation
    function stopProcessingAnimation() {
        lottieProcessingAnimation.stop();
    }

    // Set drag-and-drop events listeners on document
    let dragoverCount = 0;
    $(document).on({
        'dragenter': dragEnterHandler,
        'dragover': dragOverHandler,
        'dragleave': dragLeaveHandler,
        'drop': dropHandler
    });

    // Set paste event listener on window
    $(window).on('paste', pasteHandler);

    // Set keydown event listener on window
    $(window).on('keydown', keydownHandler);

    // Set click event listener on window
    $(window).on('click', clickHandler);

    // Handle drag enter event
    function dragEnterHandler(event) {
        event.stopPropagation();
        event.preventDefault();
        // If file is dragged onto home page, fade in drag-and-drop overlay
        if (dragoverCount === 0 && homePage.css('display') !== 'none' && modal.css('display') === 'none') {
            dragAndDropOverlay.stop().css('display', 'block').animate({
                opacity: 1
            }, 250);
        }
        dragoverCount++;
    }

    // Handle drag over event
    function dragOverHandler(event) {
        event.preventDefault();
        event.stopPropagation();
        // If file is dragged onto home page, fade in drag-and-drop overlay
        if (homePage.css('display') !== 'none' && modal.css('display') === 'none') {
            dragAndDropOverlay.css('display', 'block');
        }
    }

    // Handle drag leave event
    function dragLeaveHandler(event) {
        event.stopPropagation();
        event.preventDefault();
        dragoverCount--;
        if (dragoverCount === 0) {
            // Fade out drag-and-drop overlay
            dragAndDropOverlay.stop().animate({
                opacity: 0
            }, 250, function () {
                $(this).css('display', 'none');
            });
        }
    }

    // Handle drop event
    function dropHandler(event) {
        event.stopPropagation();
        event.preventDefault();
        dragoverCount = 0;
        // Fade out drag-and-drop overlay
        dragAndDropOverlay.stop().animate({
            opacity: 0
        }, 250, function () {
            $(this).css('display', 'none');
        });
        // Only handle drop events when home page is visible and modal is not
        if (homePage.css('display') !== 'none' && modal.css('display') === 'none') {
            // Handle image upload of dropped file
            handleImageUpload(event.originalEvent.dataTransfer.files[0]);
        }
    }

    // Handle paste event
    function pasteHandler(event) {
        // Only handle paste events when home page is visible and modal is not
        if (homePage.css('display') !== 'none' && modal.css('display') === 'none') {
            // Handle image upload of pasted file
            handleImageUpload(retrieveImageFromClipboard(event.originalEvent));
        }
    }

    // Handle keydown event
    function keydownHandler(event) {
        if (event.key === 'Escape' && modal.css('display') !== 'none') {
            // Close modal if Escape key is pressed when modal is open
            closeModal();
        } else if (event.key === 'Escape' && resultsPage.css('display') !== 'none') {
            // Go to home page if Escape key is pressed when results page is visible
            goToHomePage();
        }
    }

    // Handle click event
    function clickHandler(event) {
        if (event.target == modal[0]) {
            // Close modal if click is outside modal
            closeModal();
        }
    }

    // Retrieve image from clipboard
    function retrieveImageFromClipboard(event) {
        const items = (event.clipboardData || event.originalEvent.clipboardData).items;
        // Get first image item in clipboard
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                return items[i].getAsFile();
            }
        }
    }

    // Warn mobile device users
    if (/Android|iPhone|iPad|iPod|webOS|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
        showModal('Warning', 'You are using a mobile device. This web application is optimized for desktop use.');
    }

});