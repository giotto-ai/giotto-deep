// Unfortunately this needs to be written in oldschool JS for IE support

var eachElement = function eachElement(array, callback, scope) {
  if (scope == null) {
    scope = this;
  }

  var i = 0;
  return (function () {
    var result = [];

    while (i < array.length) {
      callback.call(scope, array[i], i);
      result.push(i++);
    }

    return result;
  })();
};

var Embed = function () {
  this.constructor = function () {
    this.elements = {
      tabs: document.querySelectorAll("#tabs .tCont"),
      actions: document.querySelectorAll("#actions a"),
      hl: document.querySelector(".hl"),
      pre: document.querySelectorAll("pre"),
      run: document.querySelector("#click-to-run .ctrCont"),
      embed: document.querySelector("#full-embed")
    };

    if (this.elements.embed){
      this.setupEvents();
      eachElement(this.elements.pre, function (element) {
        hljs.highlightBlock(element);
      });

      // reposition the highlight straight after load - no animation
      this.repositionHighlight(this.elements.actions[0], false);
      this.setHeight(this.elements.tabs[0]);
    }

    // specific to the standalone embed
    if (this.elements.run){
      self = this
      this.elements.run.addEventListener("click", function (event){
        self.loadResult();
        self.elements.run.parentNode.remove();
      });
    }
  };

  this.setupEvents = function () {
    var self = this;
    eachElement(this.elements.actions, function (action, index) {
      // switch tab
      action.addEventListener("click", function (event) {
        self.switchTab(event, action, index);
      });
    });

    // run results when it's the first tab in order
    if (
      (this.elements.actions[0] != null
        ? this.elements.actions[0].dataset.triggerType
        : undefined) === "result"
    ) {
      this.loadResult();
    }
  };

  this.setHeight = function (element) {
    var activeTab = element.getBoundingClientRect();
    var height = activeTab.height;

    window.parent.postMessage(
      [
        "embed",
        {
          slug, // set in layout
          height,
        },
      ],
      "*"
    );
  };

  this.predender = function () {
    var head = document.getElementsByTagName("head")[0];
    var prefetch = document.createElement("link");
    prefetch.setAttribute("rel", "prefetch");
    prefetch.setAttribute("href", show_src);
    head.appendChild(prefetch);

    var prerender = document.createElement("link");
    prerender.setAttribute("rel", "prerender");
    prerender.setAttribute("href", show_src);
    head.appendChild(prerender);
  };

  this.repositionHighlight = function (action, animated) {
    if (action) {
      // get size/pos data
      var position = action.getBoundingClientRect();

      // enable animation
      if (animated) {
        this.elements.hl.classList.add("animated");
      }

      this.elements.hl.style.left = position.left + "px";
      this.elements.hl.style.width = position.width + "px";
    }
  };

  this.loadResult = function (callback) {
    var iframes = document.querySelectorAll("#result iframe");
    var resultsFrame = document.createElement("iframe");
    var resultCont = document.querySelector("#result");

    // remove all previous iframes
    eachElement(iframes, function (iframe) {
      iframe.parentNode.removeChild(iframe);
    });

    var sandboxAttrs = ["allow-forms", "allow-scripts", "allow-same-origin", "allow-top-navigation-by-user-activation", "allow-downloads"];

    // don't allow alerts/confirms in fiddle listing on jsfiddle.net
    if (document.location.search.indexOf("disableModals") === -1) {
      sandboxAttrs.push("allow-modals", "allow-popups");
    }

    // apply some common props
    resultsFrame.src = show_src;
    resultsFrame.allowtransparency = true;
    resultsFrame.allowfullscreen = true;
    resultsFrame.allowpaymentrequest = true;
    resultsFrame.frameBorder = "0";
    resultsFrame.sandbox = sandboxAttrs.join(" ");
    resultsFrame.allow = "midi; geolocation; microphone; camera; display-capture; encrypted-media;";
    resultCont.appendChild(resultsFrame);

    if (callback) {
      resultsFrame.addEventListener("load", function () {
        callback.apply([this]);
      });
    }
  };

  this.switchTab = function (event, action, index) {
    event.preventDefault();
    event.stopPropagation();

    // reposition the highlight
    this.repositionHighlight(action, true);

    // set active state for tabs
    var actionParent = action.parentElement.parentElement.querySelectorAll(
      "li"
    );
    eachElement(this.elements.tabs, function (element) {
      element.classList.remove("active");
    });

    this.elements.tabs[index].classList.add("active");

    if (actionParent) {
      eachElement(actionParent, function (element) {
        element.classList.remove("active");
      });

      action.parentElement.classList.add("active");
    }

    this.setHeight(this.elements.tabs[index]);

    // load results when switcing to the resutls tab
    if (action.dataset.triggerType === "result") {
      this.loadResult();
    }
  };
};

window.addEventListener("DOMContentLoaded", function (event) {
  this.EmbedManager = new Embed();
  this.EmbedManager.constructor();
});
