const path = require("path");
const CopyPlugin = require("copy-webpack-plugin");

module.exports = (env, argv) => {
  const isDevelopment = argv.mode === "development";

  return {
    mode: isDevelopment ? "development" : "production",
    entry: isDevelopment
      ? "./src/index.js" // Development entry point
      : "./src/chatbot-widget.js", // Production entry point
    output: {
      filename: "chatbot-widget.js",
      path: path.resolve(__dirname, "dist"),
      publicPath: "/",
      library: "ChatBotWidget",
      libraryTarget: "umd",
      globalObject: "this",
      libraryExport: "default",
    },
    module: {
      rules: [
        {
          test: /\.jsx?$/,
          exclude: /node_modules/,
          loader: "babel-loader",
          options: { presets: ["@babel/preset-env", "@babel/preset-react"] },
        },
        {
          test: /\.css$/,
          use: ["style-loader", "css-loader"],
        },
        {
          test: /\.(png|jpg|jpeg|gif|svg)$/,
          type: "asset/resource",
          generator: {
            filename: "assets/[name][ext]",
          },
        },
      ],
    },
    plugins: [
      new CopyPlugin({
        patterns: [{ from: "public", to: "." }],
      }),
    ],
    resolve: { extensions: [".js", ".jsx"] },
    devServer: {
      static: {
        directory: path.join(__dirname, "dist"),
      },
      hot: true,
      port: 3000,
      open: true,
      historyApiFallback: true,
    },
    devtool: isDevelopment ? "eval-source-map" : false,
  };
};
