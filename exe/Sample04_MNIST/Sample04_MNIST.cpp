// Sample04_MNIST.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include<crtdbg.h>

#include<vector>
#include<boost/filesystem/path.hpp>

#include"Library/DataFormat/Binary/DataFormat.h"
#include"Layer/IOData/IODataLayer/IODataLayer.h"

using namespace Gravisbell;


/** データファイルをを読み込む
	@param	o_ppDataLayerTeach	教師データを格納したデータクラスの格納先ポインタアドレス
	@param	o_ppDataLayerTest	テストデータを格納したデータクラスの格納先ポインタアドレス
	@param	i_testRate			テストデータを全体の何%にするか0～1の間で設定
	@param	i_formatFilePath	フォーマット設定の入ったXMLファイルパス
	@param	i_dataFilePath		データの入ったバイナリファイルパス
	*/

Gravisbell::ErrorCode LoadSampleData(
	Layer::IOData::IIODataLayer** o_ppDataLayerTeach, Layer::IOData::IIODataLayer** o_ppDataLayerTest,
	F32 i_testRate,
	boost::filesystem::wpath i_formatFilePath,
	boost::filesystem::wpath i_dataFilePath);


int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	::_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF);
#endif

	Gravisbell::DataFormat::Binary::IDataFormat* pDataFormat_image = Gravisbell::DataFormat::Binary::CreateDataFormatFromXML(L"./DataFormat_image.xml");
	Gravisbell::DataFormat::Binary::IDataFormat* pDataFormat_label = Gravisbell::DataFormat::Binary::CreateDataFormatFromXML(L"./DataFormat_label.xml");


	// 画像の読み込み
	{
		// バッファを読み込む
		std::vector<BYTE> lpBuf;
		{
			FILE* fp = fopen("../../SampleData/MNIST/train-images.idx3-ubyte", "rb");
			if(fp == NULL)
			{
				delete pDataFormat_image;
				delete pDataFormat_label;
				return -1;
			}

			fseek(fp, 0, SEEK_END);
			U32 fileSize = ftell(fp);
			lpBuf.resize(fileSize);

			fseek(fp, 0, SEEK_SET);
			fread(&lpBuf[0], 1, fileSize, fp);

			fclose(fp);
		}

		U32 bufPos = 0;

		// ヘッダを読み込む
		bufPos = pDataFormat_image->LoadBinary(&lpBuf[0], lpBuf.size());

		printf("ImageSize : x   = %d\n", pDataFormat_image->GetBufferCountX(L"input"));
		printf("ImageSize : y   = %d\n", pDataFormat_image->GetBufferCountY(L"input"));
		printf("ImageSize : z   = %d\n", pDataFormat_image->GetBufferCountZ(L"input"));
		printf("ImageSize : ch  = %d\n", pDataFormat_image->GetBufferCountCH(L"input"));
		printf("\n");
		printf("ImageSize : cnt = %d\n", pDataFormat_image->GetDataStruct(L"input").GetDataCount());
	}

	printf("Image : %d\n", pDataFormat_image->GetVariableValue(L"images"));

	printf("Press any key to continue\n");
	getc(stdin);

	delete pDataFormat_image;
	delete pDataFormat_label;

	return 0;
}

