/*--------------------------------------------
 * FileName  : FullyConnect_Activation_FUNC.cpp
 * LayerName : 全結合ニューラルネットワークレイヤー(活性化関数付き)
 * guid      : BEBA34EC-C30C-4565-9386-56088981D2D7
 * 
 * Text      : 全結合ニューラルネットワークレイヤー.
 *           : 結合層と活性化層を一体化.
 *           : 学習時に[学習係数][ドロップアウト率]を設定できる.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"FullyConnect_Activation_FUNC.hpp"


// {BEBA34EC-C30C-4565-9386-56088981D2D7}
static const Gravisbell::GUID g_guid(0xbeba34ec, 0xc30c, 0x4565, 0x93, 0x86, 0x56, 0x08, 0x89, 0x81, 0xd2, 0xd7);

// VersionCode
static const Gravisbell::VersionCode g_version = {   1,   0,   0,   0}; 



struct StringData
{
    std::wstring name;
    std::wstring text;
};

namespace DefaultLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = L"ja";

    /** Base */
    static const StringData g_baseData = 
    {
        L"全結合ニューラルネットワークレイヤー(活性化関数付き)",
        L"全結合ニューラルネットワークレイヤー.\n結合層と活性化層を一体化.\n学習時に[学習係数][ドロップアウト率]を設定できる."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"NeuronCount",
            {
                L"ニューロン数",
                L"レイヤー内のニューロン数.\n出力バッファ数に直結する.",
            }
        },
        {
            L"DropOut",
            {
                L"ドロップアウト率",
                L"前レイヤーを無視する割合.\n1.0で前レイヤーの全出力を無視する",
            }
        },
        {
            L"ActivationType",
            {
                L"活性化関数種別",
                L"使用する活性化関数の種類を定義する",
            }
        },
        {
            L"BoolSample",
            {
                L"Bool型のサンプル",
                L"",
            }
        },
        {
            L"StringSample",
            {
                L"String型のサンプル",
                L"",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"ActivationType",
            {
                {
                    L"lenear",
                    {
                        L"リニア関数",
                        L"y = x;",
                    },
                },
                {
                    L"sigmoid",
                    {
                        L"シグモイド関数",
                        L"y = 1 / (1 + e^(-x));\n範囲 0 < y < 1\n(x=0, y=0.5)を通る",
                    },
                },
                {
                    L"sigmoid_crossEntropy",
                    {
                        L"シグモイド関数(出力レイヤー用)",
                        L"y = 1 / (1 + e^(-x));\n範囲 0 < y < 1\n(x=0, y=0.5)を通る",
                    },
                },
                {
                    L"ReLU",
                    {
                        L"ReLU（ランプ関数）",
                        L"y = max(0, x);\n範囲 0 <= y\n(x=0, y=0)を通る",
                    },
                },
                {
                    L"softmax",
                    {
                        L"SoftMax関数",
                        L"全体における自身の割合を返す関数.\ny = e^x / Σe^x;\n",
                    },
                },
                {
                    L"softmax_crossEntropy",
                    {
                        L"SoftMax関数(出力レイヤー用)",
                        L"全体における自身の割合を返す関数.\ny = e^x / Σe^x;\n",
                    },
                },
            }
        },
    };



    /** ItemData Learn <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Learn = 
    {
        {
            L"LearnCoeff",
            {
                L"学習係数",
                L"",
            }
        },
    };


    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn =
    {
    };


}


namespace CurrentLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = DefaultLanguage::g_languageCode;


    /** Base */
    static StringData g_baseData = DefaultLanguage::g_baseData;


    /** ItemData Layer Structure <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_LayerStructure = DefaultLanguage::g_lpItemData_LayerStructure;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure = DefaultLanguage::g_lpItemDataEnum_LayerStructure;


    /** ItemData Learn <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Learn;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Learn;

}



/** Acquire the layer identification code.
  * @param  o_layerCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)
{
	o_layerCode = g_guid;

	return Gravisbell::ERROR_CODE_NONE;
}

/** Get version code.
  * @param  o_versionCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode)
{
	o_versionCode = g_version;

	return Gravisbell::ERROR_CODE_NONE;
}


/** Create a layer structure setting.
  * @return If successful, new configuration information.
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSetting(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	/** Name : ニューロン数
	  * ID   : NeuronCount
	  * Text : レイヤー内のニューロン数.
	  *      : 出力バッファ数に直結する.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"NeuronCount",
			CurrentLanguage::g_lpItemData_LayerStructure[L"NeuronCount"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"NeuronCount"].text.c_str(),
			1, 65535, 200));

	/** Name : ドロップアウト率
	  * ID   : DropOut
	  * Text : 前レイヤーを無視する割合.
	  *      : 1.0で前レイヤーの全出力を無視する
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"DropOut",
			CurrentLanguage::g_lpItemData_LayerStructure[L"DropOut"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"DropOut"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.0000000000000000f));

	/** Name : 活性化関数種別
	  * ID   : ActivationType
	  * Text : 使用する活性化関数の種類を定義する
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"ActivationType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"ActivationType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"ActivationType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"lenear",
			L"リニア関数",
			L"y = x;");
		// 1
		pItemEnum->AddEnumItem(
			L"sigmoid",
			L"シグモイド関数",
			L"y = 1 / (1 + e^(-x));\n範囲 0 < y < 1\n(x=0, y=0.5)を通る");
		// 2
		pItemEnum->AddEnumItem(
			L"sigmoid_crossEntropy",
			L"シグモイド関数(出力レイヤー用)",
			L"y = 1 / (1 + e^(-x));\n範囲 0 < y < 1\n(x=0, y=0.5)を通る");
		// 3
		pItemEnum->AddEnumItem(
			L"ReLU",
			L"ReLU（ランプ関数）",
			L"y = max(0, x);\n範囲 0 <= y\n(x=0, y=0)を通る");
		// 4
		pItemEnum->AddEnumItem(
			L"softmax",
			L"SoftMax関数",
			L"全体における自身の割合を返す関数.\ny = e^x / Σe^x;\n");
		// 5
		pItemEnum->AddEnumItem(
			L"softmax_crossEntropy",
			L"SoftMax関数(出力レイヤー用)",
			L"全体における自身の割合を返す関数.\ny = e^x / Σe^x;\n");

pItemEnum->SetDefaultItem(1);
pItemEnum->SetValue(pItemEnum->GetDefault());

		pLayerConfig->AddItem(pItemEnum);
	}

	/** Name : Bool型のサンプル
	  * ID   : BoolSample
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Bool(
			L"BoolSample",
			CurrentLanguage::g_lpItemData_LayerStructure[L"BoolSample"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"BoolSample"].text.c_str(),
			true));

	/** Name : String型のサンプル
	  * ID   : StringSample
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_String(
			L"StringSample",
			CurrentLanguage::g_lpItemData_LayerStructure[L"StringSample"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"StringSample"].text.c_str(),
			L"サンプル"));

	return pLayerConfig;
}

/** Create layer structure settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLayerStructureSetting();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


/** Create a learning setting.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSetting(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	/** Name : 学習係数
	  * ID   : LearnCoeff
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"LearnCoeff",
			CurrentLanguage::g_lpItemData_Learn[L"LearnCoeff"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"LearnCoeff"].text.c_str(),
			0.0000000000000000f, 1000.0000000000000000f, 1.0000000000000000f));

	return pLayerConfig;
}

/** Create learning settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLearningSetting();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


