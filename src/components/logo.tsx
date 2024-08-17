import React from "react";
import Redirect from "@/components/icons/redirect";

const Logo = () => {
  return (
    <header
      id="logo"
      className="absolute z-[1] sm:left-[3.4rem] left-[4%] top-[5%] sm:top-6 text-[#202020]"
    >
      <h2 className="font-medium text-2xl pointer-events-none">Support.AI</h2>
      <div
        className="flex hover:underline transition text-[10px] items-center cursor-pointer justify-end ml-auto"
        onClick={() => window.open("https://uzaifm127.vercel.app", "_blank")}
      >
        <h4 className="mr-1 mt-[0.05rem]">by Uzaif</h4>
        <Redirect />
      </div>
    </header>
  );
};

export default Logo;
