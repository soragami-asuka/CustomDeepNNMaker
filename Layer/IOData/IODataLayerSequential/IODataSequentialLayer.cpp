
#include "stdafx.h"
#include "IODataSequentialLayer.h"
#include "Common/VersionCode.h"
#include "Library/SettingData/Standard.h"

#include<vector>
#include<list>


namespace Gravisbell {
namespace Layer {
namespace IOData {

// {4B4821F7-D64A-4678-BDDE-01F7BB837DD2}
static const Gravisbell::GUID g_guid(0x4b4821f7, 0xd64a, 0x4678, 0xbd, 0xde, 0x1, 0xf7, 0xbb, 0x83, 0x7d, 0xd2);

// VersionCode
static const Gravisbell::VersionCode g_version = {   1,   0,   0,   0}; 


/** レイヤーの識別コードを取得する.
  * @param  o_layerCode		格納先バッファ.
  */
extern Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)
{
	o_layerCode = g_guid;

	return Gravisbell::ERROR_CODE_NONE;
}

/** バージョンコードを取得する.
  * @param  o_versionCode	格納先バッファ.
  */
extern Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode)
{
	o_versionCode = g_version;

	return Gravisbell::ERROR_CODE_NONE;
}


/** レイヤー学習設定を作成する */
extern SettingData::Standard::IData* CreateLearningSetting(void)
{
	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(g_guid, g_version);
	if(pLayerConfig == NULL)
		return NULL;

	return pLayerConfig;
}
/** レイヤー学習設定を作成する
	@param i_lpBuffer	読み込みバッファの先頭アドレス.
	@param i_bufferSize	読み込み可能バッファのサイズ.
	@param o_useBufferSize 実際に読み込んだバッファサイズ
	@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
extern SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
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


}	// IOData
}	// Layer
}	// Gravisbell