#ifdef IODataLayer_EXPORTS
#define IODataLayer_API __declspec(dllexport)
#else
#define IODataLayer_API __declspec(dllimport)
#endif

#include"Layer/IOData/IIODataLayer.h"
#include"Common/Guiddef.h"
#include"Common/VersionCode.h"

namespace Gravisbell {
namespace Layer {
namespace IOData {

//======================================
// ���ʕ���
//======================================

/** ���C���[�̎��ʃR�[�h���擾����.
  * @param  o_layerCode		�i�[��o�b�t�@.
  * @return ���������ꍇ0. 
  */
extern IODataLayer_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode);

/** �o�[�W�����R�[�h���擾����.
  * @param  o_versionCode	�i�[��o�b�t�@.
  * @return ���������ꍇ0. 
  */
extern IODataLayer_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode);


/** ���C���[�w�K�ݒ���쐬���� */
extern IODataLayer_API SettingData::Standard::IData* CreateLearningSetting(void);
/** ���C���[�w�K�ݒ���쐬����
	@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
	@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
	@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
	@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
extern IODataLayer_API SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


//======================================
// CPU����
//======================================

/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPU(Gravisbell::IODataStruct ioDataStruct);
/** ���͐M���f�[�^���C���[���쐬����.CPU����
	@param guid			���C���[��GUID.
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPUwithGUID(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct);


//======================================
// GPU����
//======================================

/** ���͐M���f�[�^���C���[���쐬����.GPU����
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerGPU(Gravisbell::IODataStruct ioDataStruct);
/** ���͐M���f�[�^���C���[���쐬����.GPU����
	@param guid			���C���[��GUID.
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerGPUwithGUID(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct);


}	// IOData
}	// Layer
}	// Gravisbell