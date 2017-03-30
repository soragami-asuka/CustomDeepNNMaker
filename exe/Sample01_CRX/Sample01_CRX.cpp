//=============================================
// クレジットカード認証のデータを用いた実装サンプル
// 参考URL：
// ・Dropout：ディープラーニングの火付け役、単純な方法で過学習を防ぐ
//	https://deepage.net/deep_learning/2016/10/17/deeplearning_dropout.html
//
// サンプルデータURL:
// https://archive.ics.uci.edu/ml/datasets/Credit+Approval
//  データ本体
//		Data Folder > crx.data
//  データフォーマットについて
//		Data Folder > crx.names
//=============================================


#include "stdafx.h"

#include <boost/tokenizer.hpp>
#include<boost/algorithm/string.hpp>


#include"Library/DataFormat/DataFormatStringArray/DataFormat.h"


int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	::_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF);
#endif

	// フォーマットを読み込み
	auto pDataFormat = Gravisbell::DataFormat::StringArray::CreateDataFormatFromXML(L"DataFormat.xml");
	if(pDataFormat == NULL)
		return -1;

	// CSVファイルを読み込んでフォーマットに追加
	{
		// ファイルオープン
		FILE* fp = fopen("../../SampleData/crx.csv", "r");
		if(fp == NULL)
		{
			delete pDataFormat;
			return -1;
		}

		wchar_t szBuf[1024];
		while(fgetws(szBuf, sizeof(szBuf)/sizeof(wchar_t)-1, fp))
		{
			size_t len = wcslen(szBuf);
			if(szBuf[len-1] == '\n')
				szBuf[len-1] = NULL;

			// ","(カンマ)区切りで分離
			std::vector<std::wstring> lpBuf;
			boost::split(lpBuf, szBuf, boost::is_any_of(L","));

			std::vector<const wchar_t*> lpBufPointer;
			for(auto& buf : lpBuf)
				lpBufPointer.push_back(buf.c_str());


			pDataFormat->AddDataByStringArray(&lpBufPointer[0]);
		}

		// ファイルクローズ
		fclose(fp);
	}


	// 正規化
	pDataFormat->Normalize();


	// フォーマットを削除
	delete pDataFormat;


	return 0;
}

