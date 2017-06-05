#ifdef IODataSequentialLayer_EXPORTS
#define IODataSequentialLayer_API __declspec(dllexport)
#else
#define IODataSequentialLayer_API __declspec(dllimport)
#endif

#include"Layer/IOData/IIODataSequentialLayer.h"
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
extern IODataSequentialLayer_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode);

/** �o�[�W�����R�[�h���擾����.
  * @param  o_versionCode	�i�[��o�b�t�@.
  * @return ���������ꍇ0. 
  */
extern IODataSequentialLayer_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode);


/** ���C���[�w�K�ݒ���쐬���� */
extern IODataSequentialLayer_API SettingData::Standard::IData* CreateLearningSetting(void);
/** ���C���[�w�K�ݒ���쐬����
	@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
	@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
	@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
	@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
extern IODataSequentialLayer_API SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


//======================================
// CPU����
//======================================

/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerCPU(Gravisbell::IODataStruct ioDataStruct);
/** ���͐M���f�[�^���C���[���쐬����.CPU����
	@param guid			���C���[��GUID.
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerCPUwithGUID(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct);


//======================================
// GPU����
// �f�[�^���z�X�g�Ɋm��
//======================================
/** ���͐M���f�[�^���C���[���쐬����.GPU����
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPU_host(Gravisbell::IODataStruct ioDataStruct);
/** ���͐M���f�[�^���C���[���쐬����.GPU����
	@param guid			���C���[��GUID.
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPUwithGUID_host(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct);


//======================================
// GPU����
// �f�[�^���f�o�C�X�Ɋm��
//======================================
/** ���͐M���f�[�^���C���[���쐬����.GPU����
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPU_device(Gravisbell::IODataStruct ioDataStruct);
/** ���͐M���f�[�^���C���[���쐬����.GPU����
	@param guid			���C���[��GUID.
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPUwithGUID_device(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct);



}	// IOData
}	// Layer
}	// Gravisbell