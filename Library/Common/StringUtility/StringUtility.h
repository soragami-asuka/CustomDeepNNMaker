//================================
// 文字列のコード変換
//================================
#pragma once

#include<string>
#include<vector>

namespace StringUtility
{

	// 定義
	//=========================================================================
	/*!	@brief		Shift-jisの文字列を, Unicode文字列に変換する
		*	@param[in]	buf  = 変換したいShift-jisの文字列
		*	@param[out]	dest = 変換後のUnicodeの文字列を格納するバッファへのポインタ
		*	@param[in]	size = 変換後の文字列を格納するバッファのサイズ
		*	@retval		0     = 成功
		*	@retval		0以外 = 失敗
		*/
	//=========================================================================
	int ShiftjisToUnicode(const char *buf, wchar_t *dest, size_t size);

	//=========================================================================
	//	@brief		Shift-jisの文字列を, UTF8文字列に変換する
	//=========================================================================
	std::wstring ShiftjisToUnicode(std::string buf);




	//=========================================================================
	/*!	@brief		Unicodeの文字列を, Shift-jis文字列に変換する
		*	@param[in]	buf  = 変換したいUnicodeの文字列
		*	@param[out]	dest = 変換後のShift-jisの文字列を格納するバッファへのポインタ
		*	@param[in]	size = 変換後の文字列を格納するバッファのサイズ
		*	@retval		0     = 成功
		*	@retval		0以外 = 失敗
		*/
	//=========================================================================
	int UnicodeToShiftjis(const wchar_t *buf, char *dest, size_t size);

	//=========================================================================
	//	@brief		UTF8の文字列を, Shift-jis文字列に変換する
	//=========================================================================
	std::string UTF8toSJIS(const std::string& strBuf);

	//=========================================================================
	//	@brief		UTF8の文字列を, Unicode文字列に変換する
	//=========================================================================
	std::wstring UTF8toUnicode(const std::string& strBuf);

	//=========================================================================
	/*!	@brief		Unicodeの文字列を, Shift-jis文字列に変換する
		*	@param[in]	buf  = 変換したいUnicodeの文字列
		*	@param[out]	dest = 変換後のShift-jisの文字列を格納するバッファへのポインタ
		*	@param[in]	size = 変換後の文字列を格納するバッファのサイズ
		*	@retval		0     = 成功
		*	@retval		0以外 = 失敗
		*/
	//=========================================================================
	std::string UnicodeToShiftjis(const std::wstring& strWBuf);

}