/*--------------------------------------------
 * FileName  : Convolution_FUNC.cpp
 * LayerName : 畳みこみニューラルネットワーク
 * guid      : F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA
 * 
 * Text      : 畳みこみニューラルネットワーク.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"Convolution_FUNC.hpp"


// {F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA}
static const Gravisbell::GUID g_guid(0xf6662e0e, 0x1ca4, 0x4d59, 0xac, 0xca, 0xca, 0xc2, 0x9a, 0x16, 0xc0, 0xaa);

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
        L"畳みこみニューラルネットワーク",
        L"畳みこみニューラルネットワーク."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"FilterSize",
            {
                L"フィルタサイズ",
                L"畳みこみを行う入力信号数",
            }
        },
        {
            L"Input_Channel",
            {
                L"入力チャンネル数",
                L"入力チャンネル数",
            }
        },
        {
            L"Output_Channel",
            {
                L"出力チャンネル数",
                L"出力されるチャンネルの数",
            }
        },
        {
            L"Stride",
            {
                L"フィルタ移動量",
                L"畳みこみごとに移動するフィルタの移動量",
            }
        },
        {
            L"Padding",
            {
                L"パディングサイズ",
                L"",
            }
        },
        {
            L"PaddingType",
            {
                L"パディング種別",
                L"パディングを行う際の方法設定",
            }
        },
        {
            L"Initializer",
            {
                L"初期化関数",
                L"初期化関数の種類",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"PaddingType",
            {
                {
                    L"zero",
                    {
                        L"ゼロパディング",
                        L"不足分を0で埋める",
                    },
                },
            }
        },
    };



    /** ItemData Runtiime Parameter <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Runtime = 
    {
        {
            L"UpdateWeigthWithOutputVariance",
            {
                L"出力の分散を用いて重みを更新するフラグ",
                L"出力の分散を用いて重みを更新するフラグ.trueにした場合Calculate時に出力の分散が1になるまで重みを更新する.",
            }
        },
    };


    /** ItemData Runtime Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Runtime =
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


    /** ItemData Runtime <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Runtime;

    /** ItemData Runtime Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Runtime;

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
	/** Name : フィルタサイズ
	  * ID   : FilterSize
	  * Text : 畳みこみを行う入力信号数
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"FilterSize",
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : 入力チャンネル数
	  * ID   : Input_Channel
	  * Text : 入力チャンネル数
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"Input_Channel",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Input_Channel"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Input_Channel"].text.c_str(),
			1, 65535, 1));

	/** Name : 出力チャンネル数
	  * ID   : Output_Channel
	  * Text : 出力されるチャンネルの数
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"Output_Channel",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].text.c_str(),
			1, 65535, 1));

	/** Name : フィルタ移動量
	  * ID   : Stride
	  * Text : 畳みこみごとに移動するフィルタの移動量
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"Stride",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : パディングサイズ
	  * ID   : Padding
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"Padding",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Padding"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Padding"].text.c_str(),
			0, 0, 0,
			65535, 65535, 65535,
			0, 0, 0));

	/** Name : パディング種別
	  * ID   : PaddingType
	  * Text : パディングを行う際の方法設定
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"PaddingType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"zero",
			L"ゼロパディング",
			L"不足分を0で埋める");

pItemEnum->SetDefaultItem(0);
pItemEnum->SetValue(pItemEnum->GetDefault());

		pLayerConfig->AddItem(pItemEnum);
	}

	/** Name : 初期化関数
	  * ID   : Initializer
	  * Text : 初期化関数の種類
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_String(
			L"Initializer",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Initializer"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Initializer"].text.c_str(),
			L"glorot_uniform"));

	return pLayerConfig;
}

/** Create layer structure settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLayerStructureSetting();
	if(pLayerConfig == NULL)
		return NULL;

	Gravisbell::S64 useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


/** Create a runtime parameters.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameter(void)
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
	/** Name : 出力の分散を用いて重みを更新するフラグ
	  * ID   : UpdateWeigthWithOutputVariance
	  * Text : 出力の分散を用いて重みを更新するフラグ.trueにした場合Calculate時に出力の分散が1になるまで重みを更新する.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Bool(
			L"UpdateWeigthWithOutputVariance",
			CurrentLanguage::g_lpItemData_Learn[L"UpdateWeigthWithOutputVariance"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"UpdateWeigthWithOutputVariance"].text.c_str(),
			false));

	return pLayerConfig;
}

/** Create runtime parameter from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameterFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateRuntimeParameter();
	if(pLayerConfig == NULL)
		return NULL;

	Gravisbell::S64 useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


